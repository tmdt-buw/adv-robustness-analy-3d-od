import numpy as np
from utils import center_distance, velocity_l2, yaw_diff, attr_acc, scale_iou
import bisect
import pandas as pd
from scipy.spatial import cKDTree
import copy
from chamferdist import ChamferDistance
dist_func = ChamferDistance()
import torch
import copy
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox import bbox_overlaps_3d
import time

# Own imports
from metrics import Precision, Recall, ASR, score_k, DDR


class Sample(object):

    def __init__(self, sample_result, dist_thresh = 0.2, fp_thresh = 0.15, match_method="dist", innout_thresh=0.8, verbose = 0):
        """
        Initializes the Sample and preprocesses the results
        """
        self.verbose = verbose
        self.name = sample_result["name"]
        self.dist_thresh = dist_thresh
        self.fp_thresh = fp_thresh
        self.innout_thresh = innout_thresh
        self.gt_boxes = sample_result["gt_boxes"][0][0]
        self.gt_labels = sample_result["gt_labels"][0][0]
        self.result = sample_result["result"][0]
        self.adv_result = sample_result["adv_result"][0]
        t0 = time.time()
        self.data = self.prep_result(sample_result["result"][0], threshold=dist_thresh)
        self.adv_data = self.prep_result(sample_result["adv_result"][0], threshold=dist_thresh)
        if verbose >=4:
            print(f"[Sample] prepared results in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()
        self.points = sample_result["points"].float()
        self.adv_points = sample_result["adv_points"].float()
        self.points_per_obj, self.inner_points, self.outer_points, self.outer_boxes, self.inner_boxes  = self.extraction(self.points)
        self.adv_points_per_obj, self.adv_inner_points, self.adv_outer_points, self.adv_outer_boxes, self.adv_inner_boxes = self.extraction(self.adv_points)
        self.dist_to_car = self.get_dist_to_car()
        if verbose >=4:
            print(f"[Sample] Extracted Points in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()
        self.compute_metrics()
        if verbose >=4:
            print(f"[Sample] Computed metrics in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()
        self.create_table()
        if verbose >=4:
            print(f"[Sample] Created table in {time.time()-t0:.1f}s", flush = True)
        

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
        # for mAP
        pred_class_all = []
        # match_data holds the extra metrics we calculate for each match.
        match_data = {'trans_err': [],
                    'vel_err': [],
                    'scale_err': [],
                    'orient_err': [],
                    'attr_err': [],
                    'conf': [],
                    'iou': [],
                    'gt_match': [],
                    'pred_class': []}
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
                pred_class_all.append(int(sorted_labels[idx]))

                if sorted_scores[idx] > self.fp_thresh:
                    match_data['trans_err'].append(center_distance(gt_box_match, box))
                    match_data['vel_err'].append(velocity_l2(gt_box_match, box))
                    iou = scale_iou(gt_box_match, box)
                    match_data['scale_err'].append(1 - iou)
                    match_data['iou'].append(iou)

                    # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                    period = np.pi if sorted_labels[idx] == 'barrier' else 2 * np.pi #TODO: Do I really keep this? the labels I currently use are numbers, not class names
                    match_data['orient_err'].append(yaw_diff(gt_box_match, box, period=period))

                    match_data['attr_err'].append(1 - attr_acc(gt_box_match, box))
                    match_data['conf'].append(sorted_scores[idx])
                    match_data['gt_match'].append(match_gt_idx)
                    match_data['pred_class'].append(sorted_labels[idx])

            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(sorted_scores[idx])
                pred_class_all.append(int(sorted_labels[idx]))
        
        # filter out below certain confidence threshold
        idx = bisect.bisect_right(conf[::-1], self.fp_thresh) # bisect right on the reversed list
        idx = len(conf) - idx  # convert back to normal index ordering

        # Slice the lists
        fp = fp[:idx]
        tp  = tp[:idx]
        conf = conf[:idx]
        pred_class_all = pred_class_all[:idx]
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
            "pred_class_all": pred_class_all
        }

    def pr_rows_from_prep(self, sample_id: str, prep: dict, is_adv: int):
        """
            Builds rows for:
            1) pred_boxes
            2) pred_gt_candidates
            That can then be used to compute the AP and mAP
        """
        rows_pred = []
        rows_cand = []

        pred_classes = prep["pred_class"]
        scores = prep["conf"]
        pred_ids = prep["pred_id"]
        candidates = prep["candidates"]    # list-of-lists of dicts

        for (cls_id, score, cand_list, pred_id) in zip(pred_classes, scores, candidates, pred_ids):
            rows_pred.append((
                sample_id,
                is_adv,
                int(pred_id),
                int(cls_id),
                float(score),
            ))

            # store ALL candidates
            for c in cand_list:
                rows_cand.append((
                    sample_id,
                    is_adv,
                    int(pred_id),
                    int(c["gt_id"]),
                    None if c.get("dist_xy") is None else float(c["dist_xy"]),
                    None if c.get("iou_3d")  is None else float(c["iou_3d"]),
                    None if c.get("vel_err") is None else float(c["vel_err"]),
                    None if c.get("scale_err") is None else float(c["scale_err"]),
                    None if c.get("trans_err") is None else float(c["trans_err"]),
                    None if c.get("orient_err") is None else float(c["orient_err"]),
                    None if c.get("attr_err") is None else float(c["attr_err"]),
                ))
        return rows_pred, rows_cand

    def prep_mAP(self, result):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # for mAP
        pred_class_all = []
        # data needed for mAP computation
        mAP_data = {'pred_id': [],
                    'pred_class': [],
                    'candidates': [],
                    'conf': [],
                    }
        # Accumulate mAP data
        for idx, box in enumerate(sorted_boxes):
            candidates = []
            for gt_idx, gt_box in enumerate(self.gt_boxes):
                # Find closest match among ground truth boxes
                if self.gt_labels[gt_idx] == sorted_labels[idx]:
                    # dist computation
                    this_distance = center_distance(gt_box, box)
                    # IoU computation only if distance is low (saves computation time)
                    if this_distance <= 10:
                        # Box 1: pred_box
                        box_1 = box.to(device)
                        if box_1.dim() == 1:
                            box_1 = box_1.unsqueeze(0)
                        # Box 2: gt_box
                        box_2 = gt_box.to(device)
                        if box_2.dim() == 1:
                            box_2 = box_2.unsqueeze(0)
                        this_iou = bbox_overlaps_3d(box_1, box_2, coordinate='lidar')
                    else:
                        this_iou = 0
                    # store errors for each candidate
                    trans_err = center_distance(gt_box, box)
                    vel_err = velocity_l2(gt_box, box)
                    scale_err = 1 - scale_iou(gt_box, box)
                    period = np.pi if sorted_labels[idx] == 9 else 2 * np.pi # 9 is barrier
                    orient_err = yaw_diff(gt_box, box, period=period)
                    attr_err = 1 - attr_acc(gt_box, box)
                    candidates.append({
                        "gt_id": gt_idx, 
                        "dist_xy": this_distance, 
                        "iou_3d": this_iou, 
                        "vel_err": vel_err,
                        "scale_err": scale_err,
                        "trans_err": trans_err,
                        "orient_err": orient_err,
                        "attr_err": attr_err,
                        })
            # Usually threshold logic here! But I do it later so I have all information for all thresholds and compute mAP/AP with custom thresholds later
            mAP_data['pred_id'].append(idx)
            mAP_data['pred_class'].append(sorted_labels[idx])
            mAP_data['candidates'].append(candidates)
            mAP_data['conf'].append(sorted_scores[idx])
        return mAP_data

    def compute_metrics(self):
        t0 = time.time()
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
        
        if self.verbose >=4:
            print(f"[Sample] Precision and recall computed in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()

        self.asr = ASR(self.data["match_data"], self.adv_data["match_data"], threshold = self.fp_thresh)
        self.ddr = DDR(self.data["match_data"], self.adv_data["match_data"])
        if self.verbose >= 3:
            print("ASR: ", self.asr)
            print("DDR: ", self.ddr)
        if self.verbose >=4:
            print(f"[Sample] ASR and DDR computed in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()
 
        # self.chamfer_dist = 0.0
        self.chamfer_dist = dist_func(self.adv_points[:, :3][None,:,:].cuda(), self.points[:, :3][None,:,:].cuda())
        if self.verbose >= 3:
            print(f"Chamfer Distance: {self.chamferdist}")

        if self.verbose >=4:
            print(f"[Sample] Chamf dist computed in {time.time()-t0:.1f}s", flush = True)

    def create_table(self, recompute = False):
        if recompute:
            self.data = self.prep_result(self.result, threshold=self.dist_thresh)
            self.adv_data = self.prep_result(self.adv_result, threshold=self.dist_thresh)
            self.compute_metrics()
            self.points_per_obj, self.inner_points, self.outer_points, self.outer_boxes, self.inner_boxes = self.extraction(self.points)
            self.adv_points_per_obj, self.adv_inner_points, self.adv_outer_points, self.adv_outer_points, self.adv_outer_boxes, self.adv_inner_boxes = self.extraction(self.adv_points)
            self.dist_to_car = self.get_dist_to_car()
        t0 = time.time()
        table_data = {
            "sample_id": self.name, # Sample information
            "%_change": self.diff_pc_points(self.points,self.adv_points),
            "num_gt": len(self.gt_labels),
            "num_missed_gt": len(self.gt_labels)-sum(self.data["tp"]),
            "adv_num_missed_gt": len(self.gt_labels)-sum(self.adv_data["tp"]),
            "distinct_classes": len(set(self.gt_labels)),
            "num_fp": sum(self.data["fp"]), # Model data
            "num_tp": sum(self.data["tp"]),
            "adv_num_fp": sum(self.adv_data["fp"]),
            "adv_num_tp": sum(self.adv_data["tp"]),
            # "center_error_change": self.data["match_data"]["center_error_change"], # Other factors TODO: add adversarial ones
            "yaw_error_change": self.avg(self.data["match_data"]["trans_err"]),
            "scale_error_change": self.avg(self.data["match_data"]["scale_err"]),
            "adv_yaw_error_change": self.avg(self.adv_data["match_data"]["trans_err"]),
            "adv_scale_error_change": self.avg(self.adv_data["match_data"]["scale_err"]),
            # "label_error_change": self.data["match_data"]["label_error_change"],
            "asr": self.asr, # Metrics
            "ddr": self.ddr,
            "chamfer_dist": self.chamfer_dist,
            "recall": self.recall,
            "precision": self.precision,
            "adv_recall": self.adv_recall,
            "adv_precision": self.adv_precision,
            "score@0.1": self.data["score@k"][0.1], # TODO: Do I keep the non adv score@k? never used in papers, but might be nice for calculating the difference
            "score@0.5": self.data["score@k"][0.5],
            "adv_score@0.1": self.adv_data["score@k"][0.1],
            "adv_score@0.5": self.adv_data["score@k"][0.5],
        }
        self.table_entry = table_data
        if self.verbose >=4:
            print(f"[Sample] Prepared sample table in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()

        # compute per box table
        box_data_table = []
        for i in range(len(self.gt_boxes)):
            pred_class, pred_conf, adv_class, adv_conf, yaw_err, trans_err, scale_err, adv_yaw_err, adv_trans_err, adv_scale_err, iou, adv_iou = self.get_gt_info(i)
            box_data = {
                "sample_id": self.name,
                "gt_box_id": i,
                "class": self.gt_labels[i], 
                "pred_class":pred_class , 
                "conf": pred_conf, 
                "adv_pred_class": adv_class, 
                "adv_conf": adv_conf, 
                "iou": iou,
                "adv_iou": adv_iou,
                "attack_success": self.attack_success(pred_conf, adv_conf, thresh = self.fp_thresh),
                "distance_car": self.dist_to_car[i], 
                "num_points": len(self.points_per_obj[i]), 
                "adv_num_points": len(self.adv_points_per_obj[i]), 
                "%_obj_points_change": self.diff_pc(self.points_per_obj[i], self.adv_points_per_obj[i], change_threshold=0.05),
                "%_inner_points_change": self.diff_pc(self.inner_points[i], self.adv_inner_points[i], change_threshold=0.05),
                "%_outer_points_change": self.diff_pc(self.outer_points[i], self.adv_outer_points[i], change_threshold=0.05),
                "amt_inner_points": len(self.inner_points[i]),
                "amt_outer_points": len(self.outer_points[i]),
                # "occluded": False, #TODO: PROBLEM: need to look at training data (and Waymo does not have it at all)
                "yaw_err": yaw_err, 
                "adv_yaw_err": adv_yaw_err,
                "trans_err": trans_err, 
                "adv_trans_err": adv_trans_err,
                "scale_err": scale_err, #contains iou
                "adv_scale_err": adv_scale_err,
            }
            box_data_table.append(box_data)
        self.box_table = box_data_table
        if self.verbose >=4:
            print(f"[Sample] Prepared box table in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()
        
        # for mAP
        mAP_data, candidate_data = self.pr_rows_from_prep(self.name, self.prep_mAP(self.result), is_adv=0)
        adv_mAP_data, adv_candidate_data = self.pr_rows_from_prep(self.name ,self.prep_mAP(self.adv_result), is_adv=1)
        self.map_data = {"clean": mAP_data, "adv": adv_mAP_data}
        self.candidates = {"clean": candidate_data, "adv": adv_candidate_data}
        # print("map data: ", self.map_data)
        if self.verbose >=4:
            print(f"[Sample] mAP table in {time.time()-t0:.1f}s", flush = True)
        t0 = time.time()

        return table_data, box_data_table, self.map_data, self.candidates

    def avg(self, lst):
        if not lst or len(lst) == 0:
            return 0.0 
        return float(np.mean(lst))

    def attack_success(self, conf, adv_conf, thresh = 0.15):
        # Only consider attacks on detected bounding boxes
        if conf is None or conf < thresh:
            return -1 # so we can filter them out later
        if adv_conf is None:
            adv_conf = 0
        # 1 if detected before and below threshold after
        if conf > thresh and adv_conf < thresh:
            return 1
        else:
            return 0

    def get_dist_to_car(self):
        boxes = self.gt_boxes
        centers = boxes.gravity_center  # shape: (num_boxes, 3)

        # distance from ego vehicle (LiDAR origin)
        distances = torch.norm(centers, dim=1)
        return distances

    def get_gt_info(self, gt_idx):
        """
        Finds the information related to the ground truth box.
        """
        orig_confs = self.data["match_data"]["conf"]
        orig_gts = self.data["match_data"]["gt_match"]
        adv_confs = self.adv_data["match_data"]["conf"]
        adv_gts = self.adv_data["match_data"]["gt_match"]
        if gt_idx in orig_gts:
            idx = orig_gts.index(gt_idx)
            pred_class = self.data["match_data"]["pred_class"][idx]
            pred_conf = orig_confs[idx]
            yaw_err = self.data["match_data"]["orient_err"][idx]
            trans_err = self.data["match_data"]["trans_err"][idx]
            scale_err = self.data["match_data"]["scale_err"][idx]
            iou = self.data["match_data"]["iou"][idx]

        else:
            pred_class = None
            pred_conf = None
            yaw_err = None
            trans_err = None
            scale_err = None
            iou = None

        if gt_idx in adv_gts:
            idx = adv_gts.index(gt_idx)
            adv_class = self.adv_data["match_data"]["pred_class"][idx]
            adv_conf = adv_confs[idx]
            adv_yaw_err = self.adv_data["match_data"]["orient_err"][idx]
            adv_trans_err = self.adv_data["match_data"]["trans_err"][idx]
            adv_scale_err = self.adv_data["match_data"]["scale_err"][idx]
            adv_iou = self.adv_data["match_data"]["iou"][idx]
        else:
            adv_class = None
            adv_conf = None
            adv_yaw_err = None
            adv_trans_err = None
            adv_scale_err = None
            adv_iou = None

        return pred_class, pred_conf, adv_class, adv_conf, yaw_err, trans_err, scale_err, adv_yaw_err, adv_trans_err, adv_scale_err, iou, adv_iou


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
        ratio_matched = len(used_point) / max_len if max_len != 0 else 0
        percent_diff = 100 * (1-ratio_matched)
        return percent_diff

    def diff_pc_points(self, points,
                    adv_points,
                    change_threshold=0.1,
                    move_threshold=0.5):
        """
        Returns a single numpy array containing ALL changed points:
            - added points        (from adv)
            - removed points      (from original)
            - moved points        (from adv location)

        No duplicates.
        """
        if isinstance(points, torch.Tensor):
            pc = points.detach().cpu().numpy()
            adv_pc = adv_points.detach().cpu().numpy()
        else:
            pc = points
            adv_pc = adv_points

        n0 = len(pc)
        n1 = len(adv_pc)

        def greedy_match(orig_pts, adv_pts, threshold):
            if len(orig_pts) == 0 or len(adv_pts) == 0:
                return []

            tree = cKDTree(orig_pts)
            dists, nn = tree.query(adv_pts, k=1)

            order = np.argsort(dists)
            used_orig = set()
            used_adv = set()
            pairs = []

            for j in order:
                if dists[j] > threshold:
                    break
                i = int(nn[j])
                if i in used_orig:
                    continue
                used_orig.add(i)
                used_adv.add(j)
                pairs.append((i, j, float(dists[j])))

            return pairs

        # ---- PASS A: unchanged ----
        pairs_unchanged = greedy_match(pc, adv_pc, change_threshold)

        used_orig_A = set(i for i, j, d in pairs_unchanged)
        used_adv_A  = set(j for i, j, d in pairs_unchanged)

        rem_orig_idx = [i for i in range(n0) if i not in used_orig_A]
        rem_adv_idx  = [j for j in range(n1) if j not in used_adv_A]

        rem_pc = pc[rem_orig_idx]
        rem_adv_pc = adv_pc[rem_adv_idx]

        # ---- PASS B: moved ----
        pairs_moved_local = greedy_match(rem_pc, rem_adv_pc, move_threshold)

        moved_adv_indices = []
        used_orig_B = set()
        used_adv_B  = set()

        for i_loc, j_loc, d in pairs_moved_local:
            i_glob = rem_orig_idx[i_loc]
            j_glob = rem_adv_idx[j_loc]
            used_orig_B.add(i_glob)
            used_adv_B.add(j_glob)
            moved_adv_indices.append(j_glob)

        # ---- Added ----
        added_indices = [j for j in rem_adv_idx if j not in used_adv_B]

        # ---- Removed ----
        removed_indices = [i for i in rem_orig_idx if i not in used_orig_B]

        # ---- Collect ALL changed points ----
        changed_points = []

        # Added → use adv points
        if len(added_indices) > 0:
            changed_points.append(adv_pc[added_indices])

        # Moved → use adv position (new location)
        if len(moved_adv_indices) > 0:
            changed_points.append(adv_pc[moved_adv_indices])

        # Removed → use original points
        if len(removed_indices) > 0:
            changed_points.append(pc[removed_indices])

        if len(changed_points) == 0:
            return 0

        # compute percentage changed
        num_added = len(added_indices)
        # print("add: ", num_added)
        num_removed = len(removed_indices)
        # print("rem: ", num_removed)
        num_moved = len(moved_adv_indices)
        # print("mv: ", num_moved)

        num_changed = num_added + num_removed + num_moved
        # print("total: ", num_changed)
        # print("num_points: ", n0)

        percentage_changed = (num_changed / n0) * 100 if n0 > 0 else 0.0
        # print("percent changed:", percentage_changed)
        return percentage_changed
        # return np.vstack(changed_points)

    def diff_pc_list(self, points,
                    adv_points,
                    change_threshold=0.1,
                    move_threshold=0.5):
        """
        Returns a single numpy array containing ALL changed points:
            - added points        (from adv)
            - removed points      (from original)
            - moved points        (from adv location)

        No duplicates.
        """
        if isinstance(points, torch.Tensor):
            pc = points.detach().cpu().numpy()
            adv_pc = adv_points.detach().cpu().numpy()
        else:
            pc = points
            adv_pc = adv_points

        n0 = len(pc)
        n1 = len(adv_pc)

        def greedy_match(orig_pts, adv_pts, threshold):
            if len(orig_pts) == 0 or len(adv_pts) == 0:
                return []

            tree = cKDTree(orig_pts)
            dists, nn = tree.query(adv_pts, k=1)

            order = np.argsort(dists)
            used_orig = set()
            used_adv = set()
            pairs = []

            for j in order:
                if dists[j] > threshold:
                    break
                i = int(nn[j])
                if i in used_orig:
                    continue
                used_orig.add(i)
                used_adv.add(j)
                pairs.append((i, j, float(dists[j])))

            return pairs

        # ---- PASS A: unchanged ----
        pairs_unchanged = greedy_match(pc, adv_pc, change_threshold)

        used_orig_A = set(i for i, j, d in pairs_unchanged)
        used_adv_A  = set(j for i, j, d in pairs_unchanged)

        rem_orig_idx = [i for i in range(n0) if i not in used_orig_A]
        rem_adv_idx  = [j for j in range(n1) if j not in used_adv_A]

        rem_pc = pc[rem_orig_idx]
        rem_adv_pc = adv_pc[rem_adv_idx]

        # ---- PASS B: moved ----
        pairs_moved_local = greedy_match(rem_pc, rem_adv_pc, move_threshold)

        moved_adv_indices = []
        used_orig_B = set()
        used_adv_B  = set()

        for i_loc, j_loc, d in pairs_moved_local:
            i_glob = rem_orig_idx[i_loc]
            j_glob = rem_adv_idx[j_loc]
            used_orig_B.add(i_glob)
            used_adv_B.add(j_glob)
            moved_adv_indices.append(j_glob)

        # ---- Added ----
        added_indices = [j for j in rem_adv_idx if j not in used_adv_B]

        # ---- Removed ----
        removed_indices = [i for i in rem_orig_idx if i not in used_orig_B]

        # ---- Collect ALL changed points ----
        changed_points = []

        # Added → use adv points
        if len(added_indices) > 0:
            changed_points.append(adv_pc[added_indices])

        # Moved → use adv position (new location)
        if len(moved_adv_indices) > 0:
            changed_points.append(adv_pc[moved_adv_indices])

        # Removed → use original points
        if len(removed_indices) > 0:
            changed_points.append(pc[removed_indices])

        if len(changed_points) == 0:
            return np.empty((0, pc.shape[1]))

        # compute percentage changed
        num_added = len(added_indices)
        # print("add: ", num_added)
        num_removed = len(removed_indices)
        # print("rem: ", num_removed)
        num_moved = len(moved_adv_indices)
        # print("mv: ", num_moved)

        num_changed = num_added + num_removed + num_moved
        # print("total: ", num_changed)
        # print("num_points: ", n0)

        percentage_changed = (num_changed / n0) * 100 if n0 > 0 else 0.0
        # print("percent changed:", percentage_changed)
        return np.vstack(changed_points)


    def extraction(self, points):
        """
        Extracts the points of each object from the point cloud.
        Additionally distinguishes inner and outer (boundary) points.
        """
        inner_scale = self.innout_thresh
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        points_all = torch.as_tensor(points, device=device)
        points_xyz = points_all[:, :3].float()

        gt_boxes = self.gt_boxes.to(device)
        gt_boxes.tensor = gt_boxes.tensor.float()
        boxes = LiDARInstance3DBoxes(
            gt_boxes.tensor[:, :7],
            box_dim=7,
            with_yaw=True
        )

        # points inside full boxes
        mask_outer = boxes.points_in_boxes_all(points_xyz.float())  # (num_points, num_boxes)

        # Create shrunken boxes for inner points
        boxes_tensor = boxes.tensor.clone()  # (N, 7)
        # scale dx, dy, dz
        boxes_tensor[:, 3:6] *= inner_scale # shrink dx, dy, dz
        boxes_inner = LiDARInstance3DBoxes(
            boxes_tensor,
            box_dim=7,
            with_yaw=True
        ) 

        mask_inner = boxes_inner.points_in_boxes_all(points_xyz.float())

        sampled_points_per_box = []
        inner_points_per_box = []
        outer_points_per_box = []

        for b in range(mask_outer.size(1)):  # iterate over boxes
            inside_full = mask_outer[:, b] == 1
            inside_inner = mask_inner[:, b] == 1

            # All points per box
            box_points = points_all[inside_full]
            sampled_points_per_box.append(box_points)

            # Distinguish between inner and outer boxes
            inner_points = points_all[inside_inner]
            outer_points = points_all[inside_full & (~inside_inner)]

            inner_points_per_box.append(inner_points)
            outer_points_per_box.append(outer_points)

        return sampled_points_per_box, inner_points_per_box, outer_points_per_box, boxes, boxes_inner




    



