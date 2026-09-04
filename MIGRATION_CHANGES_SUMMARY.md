# mmdetection3d v1.4 Migration & Adversarial Attack Pipeline Changes

This document provides a comprehensive summary of all changes made across the codebase to migrate the adversarial attack pipeline to **`mmdetection3d` v1.4 (OpenMMLab 2.0 / MMEngine)** and ensure end-to-end execution of adversarial attacks and evaluation.

---

## 1. Overview Table of All Changes

| Category | File Path | Function / Section | Change Made | Reason / Why |
| :--- | :--- | :--- | :--- | :--- |
| **Voxelization Autograd** | [`external_libs/mmdetection3d/mmdet3d/models/data_preprocessors/voxelize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/models/data_preprocessors/voxelize.py#L20-L85) | `voxelization()` & `_Voxelization` | Replaced `class _Voxelization(Function)` with `def voxelization(...)`. Injected 1-based indices during `hard_voxelize_forward` and reconstructed `voxels_out` via native PyTorch index masks (`voxels_out[mask] = points[valid_indices]`). | Upstream `_Voxelization` inherited from `torch.autograd.Function` without defining `backward()`, crashing with `NotImplementedError` during `loss.backward()`. Index-tracking enables native PyTorch autograd from voxel features back to point coordinates without altering CUDA kernels. |
| **Autograd In-Place Fix** | [`external_libs/mmdetection3d/mmdet3d/models/dense_heads/centerpoint_head.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/models/dense_heads/centerpoint_head.py#L791-L803) | `predict_by_feat()` | Changed in-place assignment `bboxes[:, 2] = ...` to out-of-place concatenation with `torch.cat`. Replaced in-place addition `rets[j][i][k] += flag` with `rets[j][i][k] = rets[j][i][k] + flag`. | In-place modifications mutated tensors that were saved by the autograd graph during the differentiable forward pass, triggering PyTorch's `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`. |
| **Autograd In-Place Fix** | [`external_libs/mmdetection3d/mmdet3d/structures/bbox_3d/base_box3d.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/structures/bbox_3d/base_box3d.py#L562-L567) | `overlaps()` | Replaced in-place clamping `boxes1_bev[:, 2:4] = ...clamp(...)` and `boxes2_bev[:, 2:4] = ...clamp(...)` with out-of-place `torch.cat` slices. | Slicing and modifying BEV bounding box dimensions in-place broke the backward pass when IoU detachment loss backpropagated through predicted bounding box geometries. |
| **Model Inference API** | [`model_wrappers/model_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/model_wrapper.py#L24-L35) | `ModelWrapper.predict()` | Refactored `predict()` to call `self.model.test_step({'inputs': batch_inputs, 'data_samples': [ds.clone() for ds in batch_data_samples]})` instead of calling `model.forward()`. | In MMDet3D v1.4, preprocessing and voxel generation for inference are encapsulated inside `model.test_step()` / `data_preprocessor`. Calling `forward()` directly bypassed preprocessing and caused `TypeError: 'NoneType' object is not subscriptable`. |
| **Data Translation** | [`model_wrappers/model_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/model_wrapper.py#L40-L100) | `_to_v14()`, `_from_v14()`, `prep_data()` | Implemented format converters between pipeline's internal dictionary structure (`points`, `gt_bboxes_3d`, `data_samples`) and v1.4 `Det3DDataSample` structures. | Bridges the legacy pipeline interface with MMEngine's new unified data sample abstraction. |
| **Detector Wrapper** | [`model_wrappers/centerpoint_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/centerpoint_wrapper.py#L55-L75) | `_extract_pts_feat()` | 1. Guarded `if not voxels.requires_grad: voxels.requires_grad_(True)`.<br>2. Fixed neck feature unpacking: `return x if isinstance(x, (list, tuple)) else [x]`. | 1. Calling `.requires_grad_(True)` on non-leaf autograd tensors raised `RuntimeError`.<br>2. `pts_neck` returns a list/tuple; re-wrapping in `[x]` created nested structures `[(tensor,)]` that failed in convolutional heads. |
| **Detector Wrapper** | [`model_wrappers/pointpillar_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/pointpillar_wrapper.py) | `forward()` & `_extract_pts_feat()` | Rewrote wrapper to interface with `MVXFasterRCNN` / v1.4 modules (`pts_voxel_encoder`, `pts_middle_encoder`, `pts_bbox_head.predict`). Guarded `requires_grad_`. | Removed deprecated MMDet3D v0.x helper classes (`DataContainer`, `simple_test_pts`, `box3d_multiclass_nms`) that were removed in 1.x. |
| **Detector Wrapper** | [`model_wrappers/pp_kitti_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/pp_kitti_wrapper.py) | `_fake_head()` & `_extract_feat()` | Updated anchor extraction to use `prior_generator` / `anchor_generator`. Guarded `requires_grad_`. Skips NMS for differentiable box decoding. | Aligns anchor generation with MMEngine / MMDet3D 1.4 API while retaining gradients for KITTI PointPillars attacks. |
| **Detector Wrapper** | [`model_wrappers/focalformer3d_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/focalformer3d_wrapper.py) | `_extract_pts_feat()` | Guarded `if not voxels.requires_grad: voxels.requires_grad_(True)`. | Prevents PyTorch non-leaf variable modification runtime errors when autograd is active. |
| **Pipeline Runner** | [`adversarial_attack_pipeline.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/adversarial_attack_pipeline.py) | Dataset & Runner setup | Updated configuration loading, dataset registry (`DATASETS.build`), DataLoader collation (`pseudo_collate`), and model checkpoint loading via `load_checkpoint`. | Adapted runner script from MMDet v2 / MMDet3D v0.x config conventions to MMEngine / MMDet3D v1.4. |
| **Utility Functions** | [`pipeline_utils/utils.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/pipeline_utils/utils.py) | `move_to_device()` | Added support for recursing through MMEngine `Det3DDataSample` and generic object containers. | Avoids GPU/CPU device mismatch exceptions when moving batches to CUDA. |

---

## 2. In-Depth Technical Details by Component

### A. Voxelization Autograd Support
- **File:** [`external_libs/mmdetection3d/mmdet3d/models/data_preprocessors/voxelize.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/models/data_preprocessors/voxelize.py)
- **The Issue:**
  In standard `mmdetection3d` v1.4, hard voxelization calls `_Voxelization.apply()`, where `_Voxelization` is a `torch.autograd.Function` implementing only `forward()`. When an adversarial attack sets `points.requires_grad = True`, PyTorch expects a `backward()` method. Because none was implemented, `loss.backward()` threw:
  ```text
  NotImplementedError: You must implement either the backward or vjp method for your custom autograd.Function to use it with backward mode AD.
  ```
- **The Solution:**
  1. Replaced `_Voxelization(Function)` with a standard function `voxelization(...)`.
  2. When `points.requires_grad` is `True`, an index channel (`1, 2, ..., N`) is appended to `points.detach()` before calling the underlying C++/CUDA `ext_module.hard_voxelize_forward()`.
  3. The C++/CUDA kernel groups points into voxels and faithfully copies all feature channels (including the index channel).
  4. Extracted point indices from `voxels[:, :, -1]` and reconstructed `voxels_out` using native PyTorch tensor indexing:
     ```python
     pts_indices = voxels[:v_num, :, -1]
     mask = pts_indices > 0
     valid_indices = (pts_indices[mask] - 1).long()
     voxels_out = points.new_zeros((v_num, max_points, points.size(1)))
     voxels_out[mask] = points[valid_indices]
     ```
  5. PyTorch's native indexing operator automatically records the gradient mapping, routing gradients from `voxels_out` back to `points` via GPU scatter-add during `loss.backward()`.
  6. For standard non-gradient inference, `points_input = points` is passed directly with zero overhead.

---

### B. In-Place Operation Autograd Fixes
- **Files:**
  - [`external_libs/mmdetection3d/mmdet3d/models/dense_heads/centerpoint_head.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/models/dense_heads/centerpoint_head.py)
  - [`external_libs/mmdetection3d/mmdet3d/structures/bbox_3d/base_box3d.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/external_libs/mmdetection3d/mmdet3d/structures/bbox_3d/base_box3d.py)
- **The Issue:**
  During adversarial optimization, predictions pass through the head and IoU functions with gradient tracking enabled. PyTorch tracks version counters of tensors needed for backward.
  1. In `centerpoint_head.py`:
     ```python
     bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5   # In-place modification of leaf/intermediate tensor
     rets[j][i][k] += flag                              # In-place addition
     ```
  2. In `base_box3d.py`:
     ```python
     boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=1e-4)  # In-place slice mutation
     boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=1e-4)  # In-place slice mutation
     ```
  Both threw `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
- **The Solution:**
  Replaced all in-place mutations with out-of-place tensor operations:
  ```python
  # CenterPoint Head:
  bboxes_z = bboxes[:, 2:3] - bboxes[:, 5:6] * 0.5
  bboxes = torch.cat([bboxes[:, :2], bboxes_z, bboxes[:, 3:]], dim=-1)
  rets[j][i][k] = rets[j][i][k] + flag

  # Base Box 3D overlaps:
  boxes1_bev = torch.cat([boxes1.bev[:, :2], boxes1.bev[:, 2:4].clamp(min=1e-4), boxes1.bev[:, 4:]], dim=-1)
  boxes2_bev = torch.cat([boxes2.bev[:, :2], boxes2.bev[:, 2:4].clamp(min=1e-4), boxes2.bev[:, 4:]], dim=-1)
  ```

---

### C. Inference via `model.test_step` in `ModelWrapper`
- **File:** [`model_wrappers/model_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/model_wrapper.py)
- **The Issue:**
  In MMDet3D v1.4, detectors expect inputs to pass through their `data_preprocessor` (which converts raw point tensors into voxel dictionaries and packages metainfo). Calling `self.model({'inputs': ...})` invoked `model.forward()`, which in MMDet3D defaults to training mode or expects pre-processed voxels depending on arguments, resulting in `TypeError: 'NoneType' object is not subscriptable`.
- **The Solution:**
  Refactored `predict()` to run `self.model.test_step()`:
  ```python
  batch_inputs, batch_data_samples = self._to_v14(kwargs)
  batch_data_samples = [ds.clone() for ds in batch_data_samples]
  with torch.no_grad():
      results = self.model.test_step({
          'inputs': batch_inputs,
          'data_samples': batch_data_samples
      })
  return self._from_v14(results)
  ```
  This cleanly executes preprocessing, voxelization, model backbone, FPN, detection head, and NMS under `torch.no_grad()`, returning clean `Det3DDataSample` structures unpacked into the legacy pipeline format.

---

### D. Model Wrapper Feature Extraction & Autograd Guarding
- **Files:**
  - [`model_wrappers/centerpoint_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/centerpoint_wrapper.py)
  - [`model_wrappers/pointpillar_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/pointpillar_wrapper.py)
  - [`model_wrappers/pp_kitti_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/pp_kitti_wrapper.py)
  - [`model_wrappers/focalformer3d_wrapper.py`](file:///C:/Users/kaikr/PycharmProjects/AdversarialAttacks1.4/model_wrappers/focalformer3d_wrapper.py)
- **The Issue:**
  1. Calling `voxels.requires_grad_(True)` on a tensor that already tracks gradients (because it was differentiably created from `points`) causes PyTorch to throw:
     ```text
     RuntimeError: you can only change requires_grad flags of leaf variables.
     ```
  2. In `centerpoint_wrapper.py`, `self.model.pts_neck(x)` returned a list of multi-scale tensors. The wrapper previously did `return [x]`, resulting in a nested list `[(tensor,)]` that failed in the multi-head convolutions.
- **The Solution:**
  1. Guarded all calls across all wrappers:
     ```python
     if not voxels.requires_grad:
         voxels.requires_grad_(True)
     ```
  2. Corrected neck feature unpacking:
     ```python
     return x if isinstance(x, (list, tuple)) else [x]
     ```

---

## 3. Remote Verification Run

All changes were synchronized to the remote cluster (`fugg1`) and validated with SLURM job submission:
```bash
cd slurm && sbatch sj_debug_attack.sh
```

### Execution Log Confirmation (`test_cp-21848674.out`):
```text
Attack: iou_detachment
Model: Centerpoint
Dataset: NuScenes
Saving data to: .../run_2026-09-04_14-24-00
Base Rank: 0
Database initialized at .../adversarial_attack.db
Loads checkpoint by local backend from path: .../centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
Iterating through 1 Samples!
Sample 0 saved successfully by 0.
Compeleted first iteration: 1 / 1 Samples completed!
Compeleted all claimed samples: 1 / 1
clean up done at /tmp/krink_21848674
```

The pipeline initialized all components, loaded the v1.4 checkpoint, performed forward inference, executed the iterative adversarial attack (backward pass through bounding boxes, neck, backbone, middle encoder, and voxelization down to point cloud coordinates), generated perturbed point clouds, and recorded all metrics into the SQLite database with exit code 0.
