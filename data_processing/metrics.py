# Metrics implemented for my master thesis
import numpy as np
from collections import defaultdict
import torch
import copy
from mmdet3d.core.bbox import BaseInstance3DBoxes

# Own imports
from utils import max_recall_ind

# 1. Evaluation metrics based on Survey of Adversarial Methods in Autonomous driving (Gong et al. 2025)
def mAP(data, min_recall = 0.1, min_precision=0.1):
    """
    Accuracy-based metric: mean Average Precision
    mAP commonly measures how well the detected results match the ground truth labels under different IoU-thresholds
    IMPORTANT: The data is preprocessed. Whether a distance or IoU threshold is used depends on the preprocessing!
    Args:
        data (dict):    Dictionary containing the preprocessed results
        min_recall:     Clip low recalls. Default value taken from NuScenes config
        min_precision:  Clip low precision. Default value taken from NuScenes config
    """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    APs = []
    # Calculate AP for every label
    for k in data.keys():
        ap = AP_nus(data, k, min_recall, min_precision)
        APs.append(ap)
    # mean of AP
    mAP = sum(APs)/len(APs)
    return mAP


def Precision(TP,FP):
    """
    Precision.
    Precision measures the proportion of predicted positives that are actually correct.
    """
    if (TP + FP) == 0:
        return None
    return TP/(TP+FP)

def Recall(TP,FN):
    """
    Recall.
    Recall measures the proportion of true positives that were predicted correctly
    """
    if (TP + FN) == 0:
        return None
    return TP/(TP+FN)



def ASR(orig_matches, adv_matches, threshold = 0.15): #TODO decide threshold
    """
    Attack Perturbation Metric: Attack Success Rate
    ASR is a widely used metric indicating how often the generated adversarial examples or attacks cause the model output to deviate from the ground-truth. It can be defined via an error threshold or by change in the classification accuracy.
    """
    successful_attacks = 0
    valid_gt = 0

    orig_confs = orig_matches["conf"]
    orig_gts = orig_matches["gt_match"]
    adv_confs = adv_matches["conf"]
    adv_gts = adv_matches["gt_match"]

    for gt_id, orig_conf in zip(orig_gts, orig_confs):
        # print("match content ", gt_id, " ", orig_conf)
        # Only consider GTs detected in original image
        if orig_conf < threshold:
            continue

        valid_gt += 1
        if gt_id in adv_gts:
            idx = adv_gts.index(gt_id)
        else:
            idx = None
        
        # Attack succeeds if adversarial detection is missing or below threshold
        if idx is None or adv_confs[idx] < threshold:
            successful_attacks += 1

    return successful_attacks / valid_gt if valid_gt > 0 else 0.0


def PerturbationQuality():
    """
    Attack Perturbation Metric: Perturbation Quality
    Perturbation Quality is usually measured by the perturbation magnitude and the visual imperceptibility. Their ratio indicates how effective a perturbation is in power and imperceptibility to humans.
    """

# 2. Other metrics
def PKL():
    """
    TODO: Should this really be implemented? Does it make sense?
    """
    pass

def DDR(orig_matches, adv_matches, normalize=False):
    """
    Detection Degradation Ratio. (also known as Confidence Decline Rate)
    """
    ddr_values = []
    orig_confs = orig_matches["conf"]
    orig_gts = orig_matches["gt_match"]
    adv_confs = adv_matches["conf"]
    adv_gts = adv_matches["gt_match"]

    for gt_id, orig_conf in zip(orig_gts, orig_confs):
        # print("match content ", gt_id, " ", orig_conf)
        if gt_id in adv_gts:
            idx = adv_gts.index(gt_id)
        else:
            idx = None
        
        adv_conf = adv_confs[idx] if idx is not None else 0

        if normalize and orig_conf > 0:
            ddr = (orig_conf - adv_conf) / orig_conf
        else:
            ddr = orig_conf - adv_conf

        ddr_values.append(ddr)

    return sum(ddr_values) / len(ddr_values) if ddr_values else 0.0

# 3. NuScenes
TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

def NDS(data, min_recall = 0.1, min_precision = 0.1):
    """
    NuScenes Detection Score
    Consists of mAP and the five mean True Positive metrics 
       Args:
        data (dict):    Dictionary containing the preprocessed results
        min_recall:     Clip low recalls. Default value taken from NuScenes config
        min_precision:  Clip low precision. Default value taken from NuScenes config
    """
    # Calculate mAP and mTP
    MAP = mAP(data, min_recall=min_recall, min_precision=min_precision)
    MTP = mTP(data, min_recall=min_recall)
    # Calculation
    sum_TPs = 0
    for tp in MTP.values():
        sum_TPs = sum_TPs + (1-min(1,tp))

    NDS = 1/10 * (5*MAP + sum_TPs)
    return NDS

    
def AP_nus(accumulated_data, label, min_recall: float, min_precision: float) -> float:
    """ 
    Based on NuScenes code!
    Calculated average precision. 
    """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(accumulated_data[label]["precision"])
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def TP_nus(accumulated_data, label, min_recall: float, metric_name: str) -> float:
    """ 
    Based on NuScenes code!
    Calculates true positive errors. 
    """
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = max_recall_ind(accumulated_data[label]['confidence'])  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(accumulated_data[label][metric_name][first_ind: last_ind + 1]))  # +1 to include error at max recall.


def mTP(data, min_recall = 0.1):
    """
    Used in NDS!
    Calculates the mean True Positive metrics
    """
    # Sort 
    TPs = defaultdict(list,{ k:[] for k in TP_METRICS})
    for k in data.keys():
        for metric in TP_METRICS:
            tp_err = TP_nus(data, k, min_recall, metric)
            TPs[metric].append(tp_err)
    # Compute average TPs
    mTP = dict.fromkeys(TP_METRICS)
    for k in TPs.keys():
        mTP[k]= sum(TPs[k])/len(TPs[k])
    return mTP

# 4. IoU-S Attacks score@k metric
def score_k(gt_bboxes, pred_bboxes, pred_scores, score_thresholds={0.1,0.5}):
    gt_bbox = copy.deepcopy(gt_bboxes)
    iou = BaseInstance3DBoxes.overlaps(gt_bbox, pred_bboxes)
    num_gt = len(gt_bbox)
    score_at_k = {}
    for k in score_thresholds:
        score_at_k[k] = 0
    for idx in range(num_gt):
        matched_box, best_iou, conf = match_prediction(iou[idx], pred_bboxes, pred_scores)
        for k in score_thresholds:
            if best_iou == 0.0:
                score_at_k[k] += 1
            else:
                score_at_k[k] += 1 if conf < k else 0 
    for k in score_thresholds:
        score_at_k[k] = score_at_k[k]/num_gt if num_gt != 0 else 1
    return score_at_k


def match_prediction(iou_row, pred_boxes, pred_scores):
    """
    Matches a predicted Box to an iou score 
    """
    if iou_row.numel() == 0:
        return None, 0.0, 0.0  # no predictions
    best_idx = torch.argmax(iou_row)
    best_iou = iou_row[best_idx]
    if best_iou > 0.01:
        return pred_boxes[best_idx:best_idx+1], best_iou, pred_scores[best_idx]
    else:
        return None, 0.0, 0.0
    
