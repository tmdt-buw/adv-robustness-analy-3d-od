import numpy as np
import torch
import math
import sqlite3
import pickle
import os
import glob
import re
import time
    
def max_recall_ind(confidence):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

def center_distance(gt_box, pred_box) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    gt_center = gt_box[:2]
    pred_center = pred_box[:2]
    dist = torch.norm(gt_center - pred_center)
    return float(dist.item()) if isinstance(dist, torch.Tensor) else float(dist)

def velocity_l2(gt_box, pred_box) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    p = pred_box.detach().cpu().numpy() if isinstance(pred_box, torch.Tensor) else np.asarray(pred_box)
    g = gt_box.detach().cpu().numpy() if isinstance(gt_box, torch.Tensor) else np.asarray(gt_box)
    
    if len(p) >= 9 and len(g) >= 9:
        vel_p = p[7:9]
        vel_g = g[7:9]
    elif len(p) > 5 and len(g) > 5:
        vel_p = p[5:]
        vel_g = g[5:]
    else:
        return 0.0
    return float(np.linalg.norm(vel_p - vel_g))

def yaw_diff(gt_box, eval_box, period: float = 2*np.pi) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = gt_box[6].item() if isinstance(gt_box[6], torch.Tensor) else float(gt_box[6])
    yaw_pred = eval_box[6].item() if isinstance(eval_box[6], torch.Tensor) else float(eval_box[6])
    
    diff = (yaw_pred - yaw_gt + math.pi) % period - math.pi 
    return float(abs(diff))

def attr_acc(gt_box, pred_box) -> float:
    """
    TODO: Problem: I currently do not use any attributes and I don't know whether other datasets use them. I will skip this for now!
    Based on the NuScenes code, adapted to fit the code!
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc
    """
    return np.nan

def scale_iou(sample_annotation, sample_result) -> float:
    """
    Based on the NuScenes code!
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = sample_annotation[3:6]
    sr_size = sample_result[3:6]
    if isinstance(sa_size, torch.Tensor):
        sa_size = sa_size.float()
    else:
        sa_size = torch.as_tensor(sa_size, dtype=torch.float)
    if isinstance(sr_size, torch.Tensor):
        sr_size = sr_size.float()
    else:
        sr_size = torch.as_tensor(sr_size, dtype=torch.float)

    assert (sa_size > 0).all(), 'Error: sample_annotation sizes must be >0.'
    assert (sr_size > 0).all(), 'Error: sample_result sizes must be >0.'

    # Compute IoU
    min_wlh = torch.min(sa_size, sr_size)
    volume_annotation = torch.prod(sa_size)
    volume_result = torch.prod(sr_size)
    intersection = torch.prod(min_wlh)
    union = volume_annotation + volume_result - intersection
    iou = intersection / union

    return float(iou.item()) if isinstance(iou, torch.Tensor) else float(iou)

def cummean(x: np.array) -> np.array:
    """
    Copied from NuScenes code!
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

NUM_FEATURES = {
    "Kitti": 4,
    "NuScenes": 5,
    "Waymo": 4,
}

def load_pointcloud(path, num_features):
    """
    Used for .bin file loading
    """
    pts = np.fromfile(path, dtype=np.float32)
    return pts.reshape(-1, num_features)

def iter_results_db(db_path):
    """
    Iterates through all sample results in the database
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT
            task_id,
            sample_id,
            dataset,
            orig_output,
            adv_output,
            orig_pc_path,
            adv_pc_path,
            gt_boxes,
            gt_labels
        FROM attack_results
        ORDER BY task_id
    """)

    try:
        for row in cursor:
            # Loading Point clouds, alternatively use the function above if using .bin files
            with open(row["orig_pc_path"], "rb") as file:
                points = pickle.load(file)
            with open(row["adv_pc_path"], "rb") as file:
                adv_points = pickle.load(file)
            yield {
                "result": pickle.loads(row["orig_output"]),
                "dataset": row["dataset"],
                "adv_result": pickle.loads(row["adv_output"]),
                "points": points,
                "adv_points": adv_points,
                "gt_boxes": pickle.loads(row["gt_boxes"]),
                "gt_labels": pickle.loads(row["gt_labels"]),
                "name": row["sample_id"],
            }
    finally:
        conn.close()

# --- Old functions used for extracting data from pickle files ---
def iter_results_multi(base_path, mode="auto"):
    root, ext = os.path.splitext(base_path)

    if mode == "single":
        yield from iter_results(base_path)
        return

    if mode == "multi":
        i = 0
        while True:
            path = f"{root}_{i}{ext}"
            if not os.path.exists(path):
                break
            yield from iter_results(path)
            i += 1
        return

    # auto (safe + efficient): glob once
    shards = sorted(
        glob.glob(f"{root}_[0-9]*{ext}"),
        key=lambda p: int(re.search(r"_(\d+)\.pkl$", p).group(1))
    )
    if shards:
        for p in shards:
            yield from iter_results(p)
    else:
        yield from iter_results(base_path)

def iter_results(file_path):
    """
    Generator that yields one sample at a time from mixed pickle formats:
    - Old format: a single list dumped once
    - New format: many single objects appended
    """
    # print(f"[iter_results] opening {file_path}", flush=True)
    with open(file_path, "rb") as f:
        try:
            t0 = time.time()
            first = pickle.load(f)
            # print(f"[iter_results] first object loaded in {time.time()-t0:.1f}s, type={type(first)}", flush = True)

            # Case 1: old format → list of samples
            if isinstance(first, list):
                for item in first:
                    yield item
            else:
                # Case 2: new / mixed format → first object is one sample
                yield first

            # Case 3: appended samples
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

        except EOFError:
            return
