# Setting up FocalFormer3D on MMDetection3D 1.4 / MMCV 2.2

Upstream: [NVlabs/FocalFormer3D](https://github.com/NVlabs/FocalFormer3D)

The upstream release targets **mmdetection3d 0.x / 1.0.0rc with mmcv-full 1.x and
mmdet 2.x**. This guide covers running it on **mmdetection3d 1.4.0 with mmcv 2.2.0**,
which is what you are forced onto if your GPUs are newer than that stack supports, see
[VOXELIZATION_MMCV2.md §1.2](VOXELIZATION_MMCV2.md#12-why-we-cannot-stay-on-the-14x-stack-h100).

Everything described here is in [`mmdet3d_v1.4_files/`](../mmdet3d_v1.4_files/); the
layout mirrors upstream so the two can be diffed file-by-file.

> **The single most important section is [§6, the coordinate convention](#6-the-coordinate-convention-lwh-and-yaw).**
> Skip it and the model loads, runs, and reports **badly degraded mAP** with no error.

---

## 1. What you need

| Component | Version |
| :--- | :--- |
| Python | 3.11 |
| PyTorch | 2.5.1 (CUDA 12.2) |
| mmcv | **2.2.0, patched**. See [VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md) |
| mmengine | 0.10.5 |
| mmdet | 3.3.0 |
| mmdetection3d | 1.4.0 |

The mmcv patch is only required for **gradient-based attacks**. Clean evaluation of
FocalFormer3D works against stock mmcv 2.2.0.

---

## 2. Install

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.4.0
pip install -v -e .
pip install "mmdet==3.3.0" "mmengine==0.10.5"
```

Then copy the plugin in:

```bash
cp -r mmdet3d_v1.4_files/projects/mmdet3d_plugin      <mmdetection3d>/projects/
cp -r mmdet3d_v1.4_files/projects/configs/focalformer3d <mmdetection3d>/projects/configs/
```

No edits to mmdetection3d's own source are needed. FocalFormer3D is a **self-contained
plugin**, everything registers through `custom_imports`. (PillarNeSt is the opposite;
see [SETTING_UP_PILLARNEST_MMDET14.md](SETTING_UP_PILLARNEST_MMDET14.md).)

Every config carries:

```python
custom_imports = dict(imports=['projects.mmdet3d_plugin'], allow_failed_imports=False)
```

and runs from the mmdetection3d root with that root on `PYTHONPATH`.

---

## 3. What had to be refactored

Both trees use the same paths under `projects/mmdet3d_plugin/`, so every row below is a
direct diff of the same file. Line counts are `wc -l`; "changed" counts `diff` output
lines, so a moved line counts twice.

| File | upstream | ours | changed |
| :--- | ---: | ---: | ---: |
| `models/detectors/focalformer3d.py` | 374 | 2239 | **2126** |
| `models/dense_heads/focal_decoder.py` | 1689 | 1620 | **1123** |
| `models/dense_heads/deepinteraction_decoder.py` | 638 | 927 | 753 |
| `datasets/pipelines/transform_3d.py` | 922 | 826 | 655 |
| `models/backbones/swin.py` | 754 | 790 | 506 |
| `models/utils/decoder_utils.py` | 579 | 675 | 487 |
| `models/detectors/deepinteraction.py` | 257 | 501 | 403 |
| `models/necks/lss.py` | 384 | 590 | 391 |
| `models/backbones/swin_utils.py` | 517 | 551 | 374 |
| `models/utils/encoder_utils.py` | 262 | 436 | 359 |
| `models/necks/focal_encoder.py` | 222 | 440 | 348 |
| `core/bbox/assigners/hungarian_assigner.py` | 162 | 334 | 301 |
| `core/post_processing/merge_augs.py` | 184 | 360 | 270 |
| `models/utils/transformer.py` | 303 | 333 | 200 |
| `core/bbox/coders/transfusion_bbox_coder.py` | 158 | 273 | 201 |
| `models/necks/deepinteraction_encoder.py` | 84 | 228 | 200 |
| `core/hook/fading.py` | 15 | 154 | 157 |
| `models/utils/grid_mask.py` | 123 | 185 | 150 |
| `models/utils/utils.py` | 67 | 120 | 89 |
| `models/utils/open3d_utils.py` | 29 | 71 | 63 |
| `models/utils/ip_basic/depth_map_utils.py` | 287 | 327 | 54 |
| `models/utils/time_utils.py` | 77 | 104 | 45 |
| `__init__.py` | 9 | 47 | 44 |
| `models/utils/ops/bev_pool/bev_pool_op.py` | 97 | 97 | **0** |
| `models/utils/ops/bev_pool/__init__.py` | 24 | 24 | **0** |
| `models/utils/ops/locatt_ops/__init__.py` | 27 | 27 | **0** |
| `models/utils/ops/bev_pool/src/` and `models/utils/ops/locatt_ops/` CUDA and C++ sources (8 files: `.cu`, `.cuh`, `.cpp`, `.h`) | unchanged | unchanged | **0** |
| `models/detectors/__init__.py`, `necks/__init__.py`, `assigners/__init__.py`, `pipelines/__init__.py`, `core/hook/__init__.py` | none | none | **0** |

Two things worth reading off this table. The **custom CUDA ops did not change at all**.
`bev_pool` and `locatt_ops` are byte-identical, because they are self-contained
`autograd.Function`s that never touch an mm* API. Both are JIT-compiled on first
import by `torch.utils.cpp_extension.load`, so the `.cu`, `.cuh`, `.cpp` and `.h`
sources next to them have to be present or the build fails before it starts:
`bev_pool` raises `assert len(sources) > 0` and `locatt_ops` fails to compile.

Second, **the detector is where the work was**: `focalformer3d.py` is essentially a
rewrite.

### 3.1 The categories of change

**a. Registry and import paths.** `mmdet3d.models.builder` → `mmdet3d.registry`;
`@DETECTORS.register_module()` → `@MODELS.register_module()`; `mmcv.runner` →
`mmengine.runner`; `mmcv.cnn.bricks.registry` → `mmengine.registry`.
accounts for the small diffs in `utils/`.

**b. The data-flow contract.** This is most of `focalformer3d.py`. mmdet3d 1.x replaced
positional `(points, img_metas, gt_bboxes_3d, ...)` with `Det3DDataSample` objects and
split `forward_train`/`simple_test` into `loss()` / `predict()` / `_forward()` behind a
`forward(inputs, data_samples, mode)` dispatcher. Metadata that used to arrive as
`img_metas` dicts now lives in `data_sample.metainfo`, and predictions must be written
back as `InstanceData` on `pred_instances_3d`.

**c. Attention module renames.** mmcv 1.x `BaseTransformerLayer` held `attentions[0]`,
`attentions[1]`, `ffns[0]`. mmengine's replacement names them `self_attn`, `cross_attn`,
`ffn`. This is why the checkpoint needs converting, see [§5](#5-checkpoint-conversion).

**d. Coordinate conventions.** [§6](#6-the-coordinate-convention-lwh-and-yaw).

**e. Our own additions**, which are *not* ports and should be ignored if you only want
upstream running: gradient hooks, attack entry points, and the debug instrumentation in
`focal_decoder.py`.

---

## 4. Checkpoints

Upstream weights are on the [FocalFormer3D releases page](https://github.com/NVlabs/FocalFormer3D).
We used:

| Checkpoint | Reference | Converted to |
| :--- | :--- | :--- |
| `FocalFormer3D_L_ep6_mAP664_NDS709.pth` | mAP 0.664 / NDS 0.709 | `FocalFormer3D_L_ep6_converted.pth` |
| `FocalFormer3D_LC_ep6_mAP705_NDS731.pth` | mAP 0.705 / NDS 0.731 | `FocalFormer3D_LC_ep6_converted.pth` |

**Always load the converted file.** The originals load too, with a warning, and then
evaluate at a degraded score.

---

## 5. Checkpoint conversion

[`tools/convert_focalformer_ckpt.py`](../mmdet3d_v1.4_files/tools/convert_focalformer_ckpt.py)

```bash
python tools/convert_focalformer_ckpt.py \
    --src checkpoint/FocalFormer3D_L_ep6_mAP664_NDS709.pth \
    --dst checkpoint/FocalFormer3D_L_ep6_converted.pth
```

It rewrites decoder keys under `pts_bbox_head.decoder.`:

| old (mmcv 1.x) | new (mmengine) |
| :--- | :--- |
| `.attentions.0.` | `.self_attn.` |
| `.attentions.1.` | `.cross_attn.` |
| `.ffns.0.` | `.ffn.` |

Nothing else is changed, only rename was required

> **Why this cannot be skipped.** `load_checkpoint` reports missing and unexpected keys
> and then **carries on**. Skip the conversion and every decoder attention block is
> silently left at its random initialisation. The model runs and produces plausible-looking
> boxes at a much lower mAP.
>
> Verify explicitly rather than trusting the run, build the model, diff `state_dict()`
> keys against the checkpoint, and require **zero** missing and **zero** unexpected.
> `diag_focalformer_lc_keys.sh` in our tree does this; the check must run in a GPU job,
> because the plugin's `__init__.py` swallows the CUDA-dependent imports on a login node
> and you get a misleading "not in the registry" error instead.

---

## 6. The coordinate convention (l/w/h and yaw)

**This is the failure that costs the most time, because nothing reports it.** The model
loads with every key matched, inference runs, boxes come out at sensible positions with
sensible scores, and nuScenes mAP collapses.

### 6.1 Root cause

mmdetection3d changed its box convention during the v1.0 coordinate-system refactor.
mmdet3d's own [`coord_sys_tutorial.md`](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/user_guides/coord_sys_tutorial.md)
states it directly:

> For each box, the dimensions are $(w, l, h)$, and the reference direction for the yaw
> angle is the positive direction of the y axis. *(the SECOND-style convention)*
>
> - The box dimensions are $(l, w, h)$ instead of $(w, l, h)$, since $w$ corresponds to
>   $dy$ and $l$ corresponds to $dx$ in KITTI. *(mmdet3d's own convention)*

FocalFormer3D was written against the old one. Its head still **predicts `(w, l, h)` with
the old yaw reference**, while the mmdet3d 1.4 nuScenes evaluator reads `(l, w, h)` with
yaw measured from +x. The weights are not wrong, the interpretation downstream of them
is.

Because the two orderings differ only by a transpose, a wrong box is still a *plausible*
box: right centre, right height, length and width exchanged. It overlaps the ground truth
enough to look reasonable in a BEV plot and not enough to pass a nuScenes distance
threshold, which is why the symptom is bad mAP rather than a crash.

### 6.2 The fix

Applied in `FocalFormer3D.add_pred_to_datasample`, at the last point before predictions
reach the evaluator:

```python
def add_pred_to_datasample(self, batch_data_samples, bbox_results):
    # Detect dataset: NuScenes needs dim swap + yaw fix, Waymo does not
    _is_nuscenes = (hasattr(self, 'test_cfg') and self.test_cfg is not None
                    and self.test_cfg.get('pts', {}).get('dataset', '') == 'nuScenes')

    for result in bbox_results:
        bboxes_3d = result.get('bboxes_3d', ...)
        if bboxes_3d is not None and len(bboxes_3d) > 0:
            tensor = bboxes_3d.tensor
            if _is_nuscenes:
                # NuScenes: swap (w, l) -> (l, w) for evaluator
                w_v0 = tensor[:, 3].clone()
                l_v0 = tensor[:, 4].clone()
                tensor[:, 3] = l_v0
                tensor[:, 4] = w_v0
                # NuScenes: yaw convention fix (v0.x -> v1.x evaluator)
                tensor[:, 6] = -tensor[:, 6] - (np.pi / 2)
```

Three things to note:

* **`.clone()` is required.** Without it the second assignment reads the value the first
  one just overwrote, and you get `w, w, h`, every box square in plan view.
* **Both halves are needed.** Fixing dimensions without yaw, or yaw without dimensions,
  leaves mAP nearly as bad. They are one change.
* **It is gated on the dataset,** via `test_cfg.pts.dataset`. **Waymo must not get this
  transform**, its converter already emits mmdet3d-convention boxes, and applying the
  swap there breaks a working path. If you add a dataset, decide deliberately which side
  of this gate it belongs on.

### 6.3 Confirming it

Run the clean nuScenes eval and
compare against the checkpoint's own reference:

| config | reference | ours | Δ |
| :--- | :--- | :--- | ---: |
| `FocalFormer3d_L_test.py` | mAP 0.664 / NDS 0.709 | 0.6578 / 0.7042 | −0.006 / −0.005 |
| `FocalFormer3D_LC_test.py` | mAP 0.705 / NDS 0.731 | 0.7025 / 0.7282 | −0.003 / −0.003 |

Within ~0.6 of a point is the port working.

If a cleanly loaded checkpoint scores far below its reference, this is the first thing to
suspect. We have no measurement of *how* far, because we never ran full val with the fix
disabled, the only full-val numbers we have are the working ones above. Treat the
magnitude as unknown and the direction as certain.

---

## 7. Running it

```bash
cd <mmdetection3d>
export PYTHONPATH=$PWD:$PYTHONPATH

python tools/test.py \
    projects/configs/focalformer3d/FocalFormer3d_L_test.py \
    checkpoint/FocalFormer3D_L_ep6_converted.pth
```

For LiDAR+camera use `FocalFormer3D_LC_test.py`. That config did not exist upstream, the
camera blocks in `FocalFormer3D_L_v14.py` are commented out, so every value in it was
recovered from those fragments or read off the checkpoint. Its header documents each one.

---

## 8. Things to look out for

| Symptom | Cause |
| :--- | :--- |
| **mAP far below reference, checkpoint loads cleanly** | [§6](#6-the-coordinate-convention-lwh-and-yaw). First thing to suspect. |
| `FocalFormer3D is not in the mmdet3d::model registry` | No GPU. The plugin `__init__.py` catches the CUDA import failure and sets the classes to `None`. Run in a GPU job. |
| mAP a few points low, no warnings | Unconverted checkpoint ([§5](#5-checkpoint-conversion)), or `bgr_to_rgb` inverted on the LC path. |
| Camera features splatted to wrong BEV cells | `lidar2img` or `img_aug_matrix` dropped from `meta_keys`. Both affect the result and neither raises if missing. |
| LC frustum shape mismatch warning | `img_scale` given as (W, H). LiftSplatShoot is height-first: `(448, 800)`, not `(800, 448)`. |
| `'NoneType' object is not callable` in `FocalEncoderLayer` | `cam_lss=True` without `iter_bev_cam=True`. `I2P_block` is deliberately not built under LSS. |
| Registry name collisions importing BEVFusion transforms | Do not import `projects/BEVFusion` alongside the plugin. The LC pipeline uses its own `focalformer_img.py`. |

---

## See also

* [SETTING_UP_PILLARNEST_MMDET14.md](SETTING_UP_PILLARNEST_MMDET14.md), same port, opposite integration style
* [VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md), differentiable voxelization on mmcv 2.x
* [VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE.md), attacks, worked through on FocalFormer3D
