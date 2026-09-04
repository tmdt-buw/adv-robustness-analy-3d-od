# Visualization Pipeline Execution & Verification Summary

## 1. Overview
The visualization script [`sj_visualize.sh`](file:///beegfs/krink/slurm/sj_visualize.sh) / [`scripts/sj_visualize_results.sh`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/scripts/sj_visualize_results.sh) and the underlying visualizer [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) were investigated, restored, adapted for `mmdet3d` v1.4 / headless SLURM execution, and validated on remote GPU host `fugg1`.

SLURM execution finished with status **`COMPLETED`** (exit code 0) on job `21849627`, generating 3D and BEV adversarial comparison and detection overlay images.

---

## 2. Table of Changes

| File | Component / Function | Change Made | Rationale / Issue Resolved |
| :--- | :--- | :--- | :--- |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | File Restoration | Restored original 1420-line implementation from commit `d43207dbfd9fe506c0a446d81a873f97b67dbb7d`. | Commit `0e9b2c846a0fc8481729305001330581194ff055` inadvertently truncated `visualize.py` down to 21 lines, breaking all visualization routines. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | Imports & Backend | Ensured `matplotlib.use('Agg')` is invoked before pyplot import and added deprecation warning filter. | Guarantees off-screen rendering without requiring an X11 / GUI `$DISPLAY` on headless cluster nodes, and silences harmless matplotlib kwargs deprecation warnings. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | Module Imports | Wrapped imports in `try: from data_processing.sample import Sample ... except ImportError: from sample import Sample ...`. | Allows running both as a package (`python -m data_processing.visualize`) and directly as a script (`python data_processing/visualize.py`). |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `main()` | Detached and converted `scores` (`s.item()`), `gt_boxes_corners`, `pred_bbox_corners`, `pred_bbox_labels`, and `points` from CUDA tensors to NumPy arrays. | Under `mmdetection3d` v1.4, prediction bounding boxes and scores are PyTorch CUDA tensors. Attempting NumPy slicing or passing CUDA tensors to Matplotlib plotting functions causes `TypeError: can't convert cuda:0 device type tensor to numpy`. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `compare_adv()` | Converted `gt_boxes_corners`, `pred_boxes_corners`, `pred_boxes_corners_adv`, and `point_cloud` to CPU NumPy arrays; handled empty `adv_diff_points`. | Prevents CUDA device errors when overlaying clean vs. adversarial point clouds and bounding boxes; prevents index bounds error if no points changed. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `compute_dynamic_limits()` | Detached and checked tensor status on `points_xyz` and box arrays before taking `np.percentile`. | Fixed crash where calling `.cpu()` on already-converted NumPy arrays caused `AttributeError: 'numpy.ndarray' object has no attribute 'cpu'`, and passing CUDA box tensors caused `TypeError`. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `draw_box()` | Added tensor check: `if isinstance(vertices, torch.Tensor): vertices = vertices.detach().cpu().numpy()`. | Ensures 3D box wireframe coordinates are always CPU NumPy arrays before calling `pyplot_axis.plot`. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `get_point_colors*()` | Replaced deprecated `plt.cm.get_cmap(cmap)` with `plt.get_cmap(cmap)` and converted input point coordinates to NumPy. | Compatibility with Matplotlib 3.7+ and ensures norm/distance calculation runs on CPU NumPy. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | `make_dynamic_legend()` | Added fallback handling `CLASS_NAMES.get(dataset, CLASS_NAMES.get("NuScenes"))`. | Prevents `KeyError` if dataset string in SQLite database deviates in casing or formatting. |
| [`data_processing/visualize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/data_processing/visualize.py) | Object Visualization Routines | Updated `visualize_object_adv_comp`, `visualize_object`, `visualize_object_inner_outer` with empty-box guards and safe CPU tensor conversion. | Prevents crashes if a sample contains 0 ground truth objects and prevents tensor-to-numpy conversion exceptions. |
| [`scripts/sj_visualize_results.sh`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/scripts/sj_visualize_results.sh) / `slurm/sj_visualize.sh` | SLURM Environment & Paths | Configured `module load 2023a`, `source /beegfs/krink/miniconda3/bin/activate mmdet3d`, targeted debug run DB path, `--output` visualizations directory, and log directory `visualize_attack_logs`. | Matches the cluster environment, targets the latest Centerpoint debug run, and cleanly separates outputs. |

---

## 3. Remote Verification

### SLURM Run Status
- **Job ID:** `21849627`
- **Partition:** `gpushort`
- **Exit Code:** `0:0` (`COMPLETED`)

### Log Output (`visualize_attack_logs/visualize_attack-21849627.out`)
```text
/tmp/krink_21849627
Fri Sep  4 03:28:44 PM CEST 2026
copying data..
load env
Fri Sep  4 03:28:48 PM CEST 2026
starting run..
============================================================
Enhanced Adversarial Point Cloud Visualizer
============================================================
Visualizing specified Samples: ['0']
Reading data from: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/adversarial_attack.db
Color mode: depth
Theme: Dark
Output directory: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations
============================================================
Visualizing Sample:  0
  Computing depth colors...
  Saved: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations/0_bev_comparison.png
  Computing depth colors...
  Saved: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations/0_3d_depth.png
  Saved: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations/0_bev_depth.png

============================================================
Visualization complete!
Images saved to: /beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations
============================================================
clean up done at /tmp/krink_21849627
Removing /tmp/mat-debug-1694514.log
```

### Generated Artifacts
Location: `/beegfs/krink/Projects/adv-robustness-analy-3d-od/Results/debug/Centerpoint/NuScenes/iou_detachment/run_2026-09-04_14-33-49/visualizations/`
- `0_3d_depth.png` (1.3 MB) - 3D LiDAR point cloud scan with ground-truth and prediction 3D bounding boxes.
- `0_bev_comparison.png` (1.9 MB) - High-resolution BEV overlay comparing clean vs. adversarial point perturbations and prediction box shifts.
- `0_bev_depth.png` (2.2 MB) - High-resolution Bird's Eye View projection with depth coloring.
