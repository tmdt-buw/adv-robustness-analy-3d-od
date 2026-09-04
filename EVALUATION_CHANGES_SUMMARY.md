# Evaluation Pipeline Changes Summary (`sj_evaluate.sh`)

## Overview
This document details all changes implemented to enable end-to-end execution of `sj_evaluate.sh` on the remote GPU cluster following the `mmdetection3d` v1.4 migration, as well as the attack evaluation overview summary query.

Both `sj_evaluate.sh` and `sj_debug_attack.sh` complete with SLURM status `COMPLETED` (exit code `0:0`).

---

## Detailed Changes Table

| File | Location / Function | Old Behavior | New Behavior | Rationale / Why |
| :--- | :--- | :--- | :--- | :--- |
| [`data_processing/utils.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/utils.py#L42-L58) | `velocity_l2` | Passed `pred_box` / `gt_box` (CUDA tensors deserialized from SQLite) directly to `np.array(...)` | Detaches and transfers tensors to host memory via `.detach().cpu().numpy()` or `np.asarray(...)`; indexes velocity coordinates based on vector length; returns native Python `float` | Calling `np.array(...)` on a CUDA tensor raises `TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.` |
| [`data_processing/utils.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/utils.py#L27-L35) | `center_distance` | Returned `torch.norm(gt_center - pred_center)` (a `torch.Tensor` on `cuda:0`) | Extracts scalar via `.item()` and returns a native Python `float` | Returning a CUDA tensor caused downstream SQLite insertion and `np.mean` calculations in `avg()` to fail |
| [`data_processing/utils.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/utils.py#L60-L72) | `yaw_diff` | Subtracted raw index 6 elements (which were CUDA tensors) without type conversion | Casts `yaw_gt` and `yaw_pred` to Python `float` using `.item()`, returning `float(abs(diff))` | Prevents propagating 0D CUDA tensors into dictionary error fields and SQLite columns |
| [`data_processing/utils.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/utils.py#L101-L129) | `scale_iou` | Assumed CPU tensors; returned raw `torch.Tensor` IoU | Ensures `sa_size` and `sr_size` are float tensors, evaluates `(size > 0).all()`, and returns `float(iou.item())` | Prevents boolean evaluation bugs with multidimensional size tensors and guarantees a scalar float return value |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L36) | `Sample.__init__` | Kept `gt_labels` as whatever raw structure was deserialized (tensor or array) | Explicitly converts all labels to integers: `[int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in ...]` | Avoids tensor equality comparison issues during class matching and SQLite integer storage |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L69-L77) | `Sample.prep_result` | Labels and scores kept as raw tensor types | Casts scores to `float` and labels to `int` | Prevents type mismatch during sorting, matching, and metric filtering |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L129) | `Sample.prep_result` | Checked `if sorted_labels[idx] == 'barrier'` | Updated to `if sorted_labels[idx] == 9` (barrier class index in nuScenes) | `sorted_labels` contains integer class IDs, not string class names; prevents incorrect period calculation |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L255-L265) | `Sample.prep_mAP` | Assigned 2D tensor output of `bbox_overlaps_3d` directly to `this_iou` | Squeezes and extracts float scalar: `float(this_iou.squeeze().item())` | `bbox_overlaps_3d` in MMDet3D v1.4 returns a 2D tensor `(1, 1)` on GPU; storing it directly caused candidate row extraction to fail |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L315-L320) | `Sample.compute_metrics` | `ChamferDistance` returned tensor assigned directly to `self.chamfer_dist`; log printed typo `self.chamferdist` | Casts `self.chamfer_dist = float(dist_out.item())`; fixes variable name in debug print | Ensures `self.chamfer_dist` is a Python float suitable for SQLite insertion |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L340-L362) | `Sample.create_table` | Metric values (`asr`, `ddr`, `recall`, etc.) passed without scalar casting; `set(self.gt_labels)` directly | Ensures all table entries are Python `float` or `int` (`set(int(x) for x in self.gt_labels)`) | SQLite column types (`REAL`, `INTEGER`) require Python native primitives |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L370-L398) | `Sample.create_table` (box entries) | Inserted `dist_to_car[i]` and error metrics directly as tensors | Explicitly converts all dictionary fields (`distance_car`, `class`, `yaw_err`, `trans_err`, `scale_err`) to native `int` or `float` | Prevents `TypeError: Unsupported type for SQLite` in `SummaryTable.to_sqlite` |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L433-L436) | `Sample.get_dist_to_car` | Crashed if `len(self.gt_boxes) == 0` | Added guard returning empty tensor `torch.empty(0)` when no GT boxes are present | Robustness against samples with 0 annotations |
| [`data_processing/sample.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/sample.py#L489-L503) | `Sample.diff_pc` | Did not guard against 0-length point arrays before `cKDTree(pc)` | Ensures input is converted from torch tensors if needed; returns `0.0` if `max_len == 0`, `100.0` if one point cloud is empty | `scipy.spatial.cKDTree` throws `ValueError: data must not be empty` when given empty coordinate matrices |
| [`data_processing/summary_table.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/summary_table.py#L293-L360) | `SummaryTable.print_attack_overview` | Only printed final counts (`Final samples: 20`, `Final boxes: 703`) | Queries SQLite database for box-level and sample-level ASR, mean DDR, clean vs. adversarial recall, and point cloud perturbation percentage | Provides immediate post-evaluation visibility into attack effectiveness directly in SLURM console and log outputs |

---

## Verification Results

1. **`sj_evaluate.sh` Run**:
   - Job ID: `21849324`
   - Status: `COMPLETED` (Exit Code `0:0`)
   - Output summary:
     ```text
     Creating results table based on results from .../run_2026-09-04_14-33-49/adversarial_attack.db...
     Resuming: 0 samples already done
     ============================================================
     Finished Table Summary
     ============================================================
     Final samples: 20
     Final boxes: 703
     ============================================================

     ============================================================
                      ATTACK EVALUATION OVERVIEW
     ============================================================
     Overall Box-level ASR :  21.86% (73/334 detected boxes suppressed)
     Mean Sample-level ASR :  21.05%
     Mean DDR (Conf Drop)  : 0.1566
     Recall (Clean -> Adv) :  11.94% ->  11.06%
     Mean Point Perturb.   :   9.27%
     Total Ground Truths   : 703 (334 detected clean, 369 un-detected)
     ============================================================

     Done! Saved table!
     ```
   - Target database: `evaluation.db` created and populated with complete sample and per-box metrics.

2. **`sj_debug_attack.sh` Run**:
   - Job ID: `21849235`
   - Status: `COMPLETED` (Exit Code `0:0`)
   - Successfully attacked Sample 0, verified predictions and saved outputs to `adversarial_attack.db`.
