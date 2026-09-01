# Setting up PillarNeSt on MMDetection3D 1.4 / MMCV 2.2

Upstream: [WayneMao/PillarNeSt](https://github.com/WayneMao/PillarNeSt)

The upstream release targets **mmdetection3d 0.x / 1.0.0rc with mmcv-full 1.x**. This
guide covers **mmdetection3d 1.4.0 with mmcv 2.2.0**. See
[VOXELIZATION_MMCV2.md §1.2](VOXELIZATION_MMCV2.md#12-why-we-cannot-stay-on-the-14x-stack-h100)
for why that migration is not optional.

For the older stack, the original notes are still in
[SETTING_UP_PILLARNEST.md](SETTING_UP_PILLARNEST.md). This file supersedes them for
mmdet3d 1.4.

> **Read [§6, the coordinate convention](#6-the-coordinate-convention-lwh-and-yaw) before
> you evaluate anything.** Get it wrong and the model loads cleanly and scores far below
> its reference, with no error.

---

## 1. Integration style: not a plugin

The important structural difference from
[FocalFormer3D](SETTING_UP_FOCALFORMER3D_MMDET14.md): upstream PillarNeSt ships as a
plugin under `projects/mmdet3d_plugin/`, but **we integrated it into `mmdet3d/` core
instead**, and its config builds the stock `CenterPoint` detector with PillarNeSt
components swapped in:

```python
model = dict(
    type='CenterPoint',                              # stock mmdet3d detector
    pts_voxel_encoder=dict(type='PillarNestHeightFeatureNet', ...),
    pts_middle_encoder=dict(type='PointPillarsScatter', ...),   # stock
    pts_backbone=dict(type='PillarNestConvNeXt', arch='large', ...),
    pts_neck=dict(type='SECONDFPN', ...),                       # stock
    pts_bbox_head=dict(type='PillarNestCenterHead', ...),
)
```

(That block is from `pillarnest_large_mininus.py`. Configs differ in what they override,
`pillarnest_large_clean.py`, for instance, `_delete_`s the head back to stock `CenterHead`
while keeping `PillarNestBBoxCoder`. Read the config you are actually running.)

PillarNeSt is a set of *components*, not a new detector thus the detector logic it needs is
already `CenterPoint`. Putting the components in core lets them compose with stock
mmdet3d modules without a plugin namespace in between, and avoids the registry-name
collisions the upstream plugin hits (its files are named `centerpoint_bbox_coders.py`,
`centerpoint_plus_head.py`, and so on, which clash with mmdet3d's own).

The cost is that **this patches mmdetection3d itself**, so the install is a file copy plus
four `__init__.py` edits rather than a `custom_imports` line. Every file we add is
prefixed `pillarnest_` so nothing overwrites a stock module.

---

## 2. What you need

| Component | Version |
| :--- | :--- |
| Python | 3.11 |
| PyTorch | 2.5.1 (CUDA 12.2) |
| mmcv | 2.2.0, [patched](VOXELIZATION_MMCV2.md) only if you need attack gradients |
| mmengine | 0.10.5 |
| mmdet | 3.3.0 |
| mmdetection3d | 1.4.0 |

---

## 3. Install

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v1.4.0

cp ../mmdet3d_v1.4_files/mmdet3d/models/backbones/pillarnest_convnext.py             mmdet3d/models/backbones/
cp ../mmdet3d_v1.4_files/mmdet3d/models/dense_heads/pillarnest_center_head.py        mmdet3d/models/dense_heads/
cp ../mmdet3d_v1.4_files/mmdet3d/models/task_modules/coders/pillarnest_bbox_coder.py mmdet3d/models/task_modules/coders/
cp ../mmdet3d_v1.4_files/mmdet3d/models/voxel_encoders/pillarnest_pillar_encoder.py  mmdet3d/models/voxel_encoders/
cp ../mmdet3d_v1.4_files/mmdet3d/models/voxel_encoders/pillarnest_utils.py           mmdet3d/models/voxel_encoders/
cp -r ../mmdet3d_v1.4_files/configs/pillarnest                                       configs/

pip install -v -e .
```

Please note that you need to change to mmcv_maximum_version = '2.3.0' inside mmdet3d/__init__.py. 



### 3.1 Register them

Five files, four `__init__.py`. Nothing registers without this, and the failure is a
config error that looks unrelated (`PillarNestConvNeXt is not in the mmdet3d::model
registry`).

```python
# mmdet3d/models/backbones/__init__.py
from .pillarnest_convnext import PillarNestConvNeXt
__all__ = [..., 'PillarNestConvNeXt']

# mmdet3d/models/dense_heads/__init__.py
from .pillarnest_center_head import PillarNestCenterHead
__all__ = [..., 'PillarNestCenterHead']

# mmdet3d/models/voxel_encoders/__init__.py
from .pillarnest_pillar_encoder import (PillarNestFeatureNet,
                                        PillarNestSEFeatureNet,
                                        PillarNestHeightFeatureNet)
from .pillarnest_utils import (PFNLayer, SEPFNLayer, SEPFNLayerV2,
                               ChannelAttention, ChannelAttentionV2,
                               get_paddings_indicator)
__all__ = [..., 'PillarNestFeatureNet', 'PillarNestSEFeatureNet',
           'PillarNestHeightFeatureNet', 'PFNLayer', ...]

# mmdet3d/models/task_modules/coders/__init__.py
from .pillarnest_bbox_coder import PillarNestBBoxCoder
__all__ = [..., 'PillarNestBBoxCoder']
```

---

## 4. What had to be refactored

Upstream's plugin files map one-to-one onto ours. "Changed" counts `diff` output lines,
so a moved line counts twice.

| upstream (`projects/mmdet3d_plugin/`) | ours (`mmdet3d/`) | up | ours | changed |
| :--- | :--- | ---: | ---: | ---: |
| `models/dense_heads/centerpoint_plus_head.py` | `models/dense_heads/pillarnest_center_head.py` | 338 | 398 | 522 |
| `models/backbones/convnext_pc.py` | `models/backbones/pillarnest_convnext.py` | 262 | 402 | 420 |
| `models/voxel_encoders/pillar_encoder.py` | `models/voxel_encoders/pillarnest_pillar_encoder.py` | 328 | 337 | 397 |
| `core/bbox/coder/centerpoint_bbox_coders.py` | `models/task_modules/coders/pillarnest_bbox_coder.py` | 244 | 203 | 319 |
| `models/voxel_encoders/utils.py` | `models/voxel_encoders/pillarnest_utils.py` | 244 | 302 | 309 |

Note the **path change is itself part of the port**: mmdet3d 1.x moved bbox coders from
`core/bbox/coder/` to `models/task_modules/coders/`.

### 4.1 Categories of change

**a. Registry and imports.** `mmdet3d.models.builder` → `mmdet3d.registry`;
`@BACKBONES` / `@HEADS` / `@VOXEL_ENCODERS` all collapse to `@MODELS`; `@BBOX_CODERS` →
`@TASK_UTILS`. `mmcv.runner.BaseModule` → `mmengine.model.BaseModule`;
`mmcv.cnn.build_norm_layer` stays but moves under `mmengine`.

**b. Head API.** `PillarNestCenterHead` had to move to mmdet3d 1.x's
`loss()` / `predict()` split and consume `Det3DDataSample`, matching how the stock
`CenterHead` it sits beside now works.

**c. Type annotations and config typing.** mmdet3d 1.4 modules are annotated
(`Optional[Dict]`, `List[int]`) and mmengine validates more aggressively at build time;
several upstream defaults were implicit and had to be made explicit.

**d. Coordinate conventions.** [§6](#6-the-coordinate-convention-lwh-and-yaw).

**e. Our own additions**, not part of the port: the `debug` / `debug_max_print`
instrumentation threaded through every component, and the three `legacy_*` compatibility
flags.

---

## 5. Checkpoints and conversion

Upstream weights (nuScenes **val**), from the
[PillarNeSt README](https://github.com/WayneMao/PillarNeSt#results):

| Model | mAP | NDS |
| :--- | ---: | ---: |
| PillarNeSt-Base | 63.2% | 69.2% |
| PillarNeSt-Large | 64.3% | 70.4% |

[`tools/convert_pillarnest_ckpt_to_mmdet14.py`](../mmdet3d_v1.4_files/tools/convert_pillarnest_ckpt_to_mmdet14.py)

```bash
python tools/convert_pillarnest_ckpt_to_mmdet14.py \
    --in_ckpt  checkpoint/pillarnest_large.pth \
    --out_ckpt checkpoint/pillarnest_large_v14.pth \
    --config   configs/pillarnest/pillarnest_large_clean.py
```

It strips wrapper prefixes (`module.`, `model.`, `net.`, `detector.`) and rewrites
component prefixes to the mmdet3d 1.4 `pts_*` names:

| old | new |
| :--- | :--- |
| `voxel_encoder.` / `reader.` | `pts_voxel_encoder.` |
| `middle_encoder.` | `pts_middle_encoder.` |
| `backbone.` | `pts_backbone.` |
| `neck.` | `pts_neck.` |
| `bbox_head.` | `pts_bbox_head.` |

Keys already starting with `pts_` pass through, so the script is idempotent.

**Pass `--config`.** It is optional and you should use it anyway: the script then builds
the target model and reports unexpected keys, shape mismatches and missing keys before
you spend a run finding out. `--drop_shape_mismatch` keeps only compatible tensors, use
it to *diagnose*, not to make a mismatch go away, since a dropped tensor is a randomly
initialised layer.

---

## 6. The coordinate convention (l/w/h and yaw)

Same root cause as FocalFormer3D. The full explanation, with the
citation to mmdet3d's own coordinate tutorial, is in
[SETTING_UP_FOCALFORMER3D_MMDET14.md §6.1](SETTING_UP_FOCALFORMER3D_MMDET14.md#61-root-cause).
In short: PillarNeSt's head predicts SECOND-style **`(w, l, h)` with yaw referenced to
+y**, while mmdet3d 1.4's nuScenes evaluator expects **`(l, w, h)` with yaw from +x**.

The weights are fine. Only the interpretation downstream of them is wrong which is why
you will notice collapsed mAP values, and why a BEV plot still looks
approximately right.

### 6.1 config flags

Where FocalFormer3D applies the transform in `add_pred_to_datasample`, PillarNeSt exposes
it as **config flags**, because the box decode already lives in one place. All three
default to `False`.

**On the bbox coder**: these are the evaluator-facing ones, in
`PillarNestBBoxCoder.decode`:

```python
bbox_coder=dict(
    type='PillarNestBBoxCoder',
    legacy_dim_swap=True,       # dim = dim[..., [1, 0, 2]]      (w,l,h) -> (l,w,h)
    legacy_yaw_transform=True,  # rot = -rot - (np.pi / 2)
    ...
)
```

**On the head**: `legacy_iou_transform` is a *different* thing despite the similar name.
It sits in the helper that decodes predictions for **IoU computation in the loss**, instead of on
the path to the evaluator:

```python
if self.legacy_iou_transform:
    dims_log = dims_log[:, :, [1, 0, 2]]
    yaw = -torch.atan2(rot_sin, rot_cos) - (np.pi / 2)
else:
    yaw = torch.atan2(rot_sin, rot_cos)
```

`-rot - π/2` is character-for-character the transform FocalFormer3D applies and it is the
mmdet3d v0.x → v1.x yaw conversion.

---

## 7. Running it

```bash
cd <mmdetection3d>
python tools/test.py \
    configs/pillarnest/pillarnest_large_clean.py \
    checkpoint/pillarnest_large.pth
```

Configs in [`mmdet3d_v1.4_files/configs/pillarnest/`](../mmdet3d_v1.4_files/configs/pillarnest/):

| config | purpose |
| :--- | :--- |
| `pillarnest_large_clean.py` | nuScenes val, clean eval, start here (see note) |
| `pillarnest_large_adv.py` | adversarial eval |
| `pillarnest_large_mininus.py` | val-as-train, for gradient extraction and quick runs |
| `pillarnest_kitti_adv.py`, `pillarnest_waymo_adv.py` | KITTI / Waymo |

> **`pillarnest_large_clean.py` carried a syntax error** and could not be parsed at all:
> commenting out `#iou=(1, 2)),` removed the comma that closed the `common_heads` dict, so
> `share_conv_channel=64` followed a `)` with nothing between them. The copy in
> `mmdet3d_v1.4_files/` has the comma restored. If you are working from our mmdetection3d
> tree rather than this one, apply the same one-character fix there.
>
> These configs also carry `debug=True` with `debug_max_print` set on several components,
> which prints diagnostics for the first N calls. Harmless, but set `debug=False` for
> timed runs.

**KITTI**, carried over from the original notes: it does not use the CenterPoint-plus
head, which is not KITTI-compatible. It uses the standard CenterHead
([fixed](https://github.com/open-mmlab/mmdetection3d/pull/924) for KITTI) and is expected
to score worse than the nuScenes numbers above.

---

## 8. Things to look out for

| Symptom | Cause |
| :--- | :--- |
| **mAP far below reference, checkpoint loads cleanly** | Coordinate convention. Try `legacy_dim_swap` / `legacy_yaw_transform` both ways, [§6.1](#61-the-controls-config-flags-not-code) explains why neither state is universally right. |
| `PillarNestConvNeXt is not in the mmdet3d::model registry` | `__init__.py` edits not applied ([§3.1](#31-register-them)), or mmdet3d not reinstalled after copying. |
| Many unexpected keys on load | Checkpoint not converted ([§5](#5-checkpoints-and-conversion)). Run with `--config` to see the diff. |
| Shape mismatch on `pts_backbone` | `arch` mismatch, `pillarnest_large.pth` needs `arch='large'`, not `'base'`. |
| IoU scores implausible, mAP mediocre but not zero | `legacy_iou_transform` not set on the head. |
| Slow first iterations, heavy stdout | `debug=True` in the shipped configs. |

---

## See also

* [SETTING_UP_FOCALFORMER3D_MMDET14.md](SETTING_UP_FOCALFORMER3D_MMDET14.md), same port, plugin-style integration
* [SETTING_UP_PILLARNEST.md](SETTING_UP_PILLARNEST.md), the original mmdet3d 1.0.0rc notes
* [VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md), differentiable voxelization on mmcv 2.x
