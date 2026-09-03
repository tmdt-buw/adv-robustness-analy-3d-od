import numpy as np
from utils import center_distance, velocity_l2, yaw_diff, attr_acc, scale_iou
import bisect
import pandas as pd
from scipy.spatial import cKDTree
import copy
from chamferdist import ChamferDistance
dist_func = ChamferDistance()

# Own imports
from metrics import Precision, Recall, ASR, score_k, DDR
from sample import Sample

class BoxSample(Sample):

    def __init__(self, sample_result, dist_thresh = 0.2, fp_thresh = 0.15, match_method="dist", verbose = 0):
        """
        Initializes the Sample and preprocesses the results
        """
        self.verbose = verbose
        self.name = sample_result["name"]
        self.dist_thresh = dist_thresh
        self.fp_thresh = fp_thresh
        self.gt_boxes = sample_result["gt_boxes"][0][0]
        self.gt_labels = sample_result["gt_labels"][0][0]
        self.result = sample_result["result"][0]
        self.adv_result = sample_result["adv_result"][0]
        self.data = self.prep_result(sample_result["result"][0], threshold=dist_thresh)
        self.adv_data = self.prep_result(sample_result["adv_result"][0], threshold=dist_thresh)
        self.points = sample_result["points"]
        self.adv_points = sample_result["adv_points"]
        self.points_per_obj = self.extraction(self.result)
        self.adv_points_per_obj = self.extraction(self.adv_result)
        self.compute_metrics()
        self.create_table()
        

    def prep_result(self, result, threshold=0.2, match_method="dist"):
        """
        Extracts the fp, tp and their confidences from the data
        """
        # Extract results
        pred_boxes = result["pts_bbox"]["boxes_3d"]
        pred_labels = result["pts_bbox"]["labels_3d"]
        scores = [s.item() for s in result["pts_bbox"]["scores_3d"]]
        # Compute score@k
        score_at_k = score_k(self.gt_boxes, pred_boxes, scores)
        # sort after highest confidence
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Reorder boxes and scores according to sorted indices
        sorted_boxes = [list(pred_boxes[i])[0] for i in sorted_indices]
        sorted_labels = [pred_labels[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        # Accumulators
        tp = []
        fp = []
        conf = []
        # match_data holds the extra metrics we calculate for each match.
        match_data = {'trans_err': [],
                    'vel_err': [],
                    'scale_err': [],
                    'orient_err': [],
                    'attr_err': [],
                    'conf': [],
                    'gt_match': []}
        # Match and accumulate match data
        taken = set()
        for idx, box in enumerate(sorted_boxes):
            min_dist = np.inf
            match_gt_idx = None
            gt_box_match = None
            # Next steps: compare to gt_boxes using distance function or iou, find closest ones (take the highest confidence that is in threshold)
            # For now the NuScenes way: using distance, not IoU
            for gt_idx, gt_box in enumerate(self.gt_boxes):
                # Find closest match among ground truth boxes
                if self.gt_labels[gt_idx] == sorted_labels[idx] and not gt_idx in taken:
                    this_distance = center_distance(gt_box, box)
                    # print("Distance: ", this_distance)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx
                        gt_box_match = gt_box
            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < threshold
            if is_match:
                taken.add(match_gt_idx)
                # print("Found match for: ", match_gt_idx)
                #  Update tp, fp and confs.
                tp.append(1)
                fp.append(0)
                conf.append(sorted_scores[idx])

                if sorted_scores[idx] > self.fp_thresh:
                    match_data['trans_err'].append(center_distance(gt_box_match, box))
                    match_data['vel_err'].append(velocity_l2(gt_box_match, box))
                    match_data['scale_err'].append(1 - scale_iou(gt_box_match, box))

                    # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                    period = np.pi if sorted_labels[idx] == 'barrier' else 2 * np.pi #TODO: Do I really keep this? the labels I currently use are numbers, not class names
                    match_data['orient_err'].append(yaw_diff(gt_box_match, box, period=period))

                    match_data['attr_err'].append(1 - attr_acc(gt_box_match, box))
                    match_data['conf'].append(sorted_scores[idx])
                    match_data['gt_match'].append(match_gt_idx)

            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(sorted_scores[idx])
        
        # filter out below certain confidence threshold
        idx = bisect.bisect_right(conf[::-1], self.fp_thresh) # bisect right on the reversed list
        idx = len(conf) - idx  # convert back to normal index ordering

        # Slice the lists
        fp = fp[:idx]
        tp  = tp[:idx]
        conf = conf[:idx]
        if self.verbose >= 3:
            print("FP: ", sum(fp))
            print("TP: ", sum(tp))
            print("conf: ", conf)
            print("Match data: ", match_data)

        return {
            "fp": fp,
            "tp": tp,
            "conf": conf,
            "match_data": match_data,
            "score@k": score_at_k,
        }

    def compute_metrics(self):
        fp = sum(self.data["fp"])
        tp = sum(self.data["tp"])
        self.recall = Recall(tp,fp)
        self.precision = Precision(tp,fp)
        if self.verbose >= 3:
            print(f"Precision: {self.precision}, Recall {self.recall}")

        fp = sum(self.adv_data["fp"])
        tp = sum(self.adv_data["tp"])
        self.adv_recall = Recall(tp,fp)
        self.adv_precision = Precision(tp,fp)
        if self.verbose >= 3:
            print(f"Adv Precision: {self.adv_precision}, Adv Recall {self.adv_recall}")

        self.asr = ASR(self.data["match_data"], self.adv_data["match_data"])
        self.ddr = DDR(self.data["match_data"], self.adv_data["match_data"])
        if self.verbose >= 3:
            print("ASR: ", self.asr)
            print("DDR: ", self.ddr)
 
        self.chamfer_dist = dist_func(self.adv_points[:, :3][None,:,:], self.points[:, :3][None,:,:])
        if self.verbose >= 3:
            print(f"Chamfer Distance: {self.chamferdist}")

    def create_table(self, recompute = False):
        if recompute:
            self.data = prep_result(self.result, threshold=self.dist_thresh)
            self.adv_data = prep_result(self.adv_result, threshold=self.dist_thresh)
            self.compute_metrics()
            self.points_per_obj = self.extraction(self.result)
            self.adv_points_per_obj = self.extraction(self.adv_result)

        table_data = {
            "sample_id": self.name, # Sample information
            "%_change": self.diff_pc(self.points,self.adv_points),
            "num_gt": len(self.gt_labels),
            "num_missed_gt": len(self.gt_labels)-sum(self.data["tp"]),
            "adv_num_missed_gt": len(self.gt_labels)-sum(self.adv_data["tp"]),
            "distinct_classes": len(set(self.gt_labels)),
            "num_fp": sum(self.data["fp"]), # Model data
            "num_tp": sum(self.data["tp"]),
            "adv_num_fp": sum(self.adv_data["fp"]),
            "adv_num_tp": sum(self.adv_data["tp"]),
            # "center_error_change": self.data["match_data"]["center_error_change"], # Other factors TODO: add adversarial ones
            "yaw_error_change": self.data["match_data"]["trans_err"],
            "scale_error_change": self.data["match_data"]["scale_err"],
            "adv_yaw_error_change": self.adv_data["match_data"]["trans_err"],
            "adv_scale_error_change": self.adv_data["match_data"]["scale_err"],
            # "label_error_change": self.data["match_data"]["label_error_change"],
            "occluded": False, #TODO
            "asr": self.asr, # Metrics
            "ddr": self.ddr,
            "chamfer_dist": self.chamfer_dist,
            "recall": self.recall,
            "precision": self.precision,
            "adv_recall": self.adv_recall,
            "adv_precision": self.adv_recall,
            "score@0.1": self.data["score@k"][0.1], # TODO: Do I keep the non adv score@k? never used in papers, but might be nice for calculating the difference
            "score@0.5": self.data["score@k"][0.5],
            "adv_score@0.1": self.adv_data["score@k"][0.1],
            "adv_score@0.5": self.adv_data["score@k"][0.5],
        }
        self.table_entry = table_data
        return table_data

    def diff_pc(self, points, adv_points, change_threshold = 0.1):
        """
        finds the number of unchanged points and returns 100% - %unchanged
        """
        pc = points.detach().cpu().numpy()
        adv_pc = adv_points.detach().cpu().numpy()
        max_len = len(pc) if len(pc) >= len(adv_pc) else len(adv_pc)
        # match points to find unchanged ones
        tree = cKDTree(pc)
        
        # nearest neighbor distance from each adv point to original
        dists, nn_idx = tree.query(adv_pc, k=1)

        # sort matches by distance (closest first)
        order = np.argsort(dists)

        used_point = set()
        
        for i in order: # TODO: fix matching, does not seem to be unique
            dist = dists[i]
            if dist <= change_threshold and i not in used_point:
                # accept the match
                used_point.add(i)
        ratio_matched = len(used_point) / max_len
        percent_diff = 100 * (1-ratio_matched)
        return percent_diff

    def extraction(self, result):
        """
        Extracts the points of each object from the point cloud
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        points_all = torch.as_tensor(self.points, device=device)
        points_xyz = points_all[:, :3]


        scores = result['pts_bbox']['scores_3d']  # (B,)
        pred_boxes = result['pts_bbox']['boxes_3d']    # LiDARInstance3DBoxes

        lidar_boxes = LiDARInstance3DBoxes(
            pred_boxes.tensor[:, :7],
            box_dim=7,
            with_yaw=True
        )

        # --- point-in-box ---
        mask = lidar_boxes.points_in_boxes_all(points_xyz)  # (N, K)
        sampled_points_per_box = []
        for b in range(mask.size(1)):
            box_points = points_all[mask[:, b] >= 0]
            sampled_points_per_box.append(box_points)
        print("points per boxes: ", sampled_points_per_box)
        return sampled_points_per_box


    



