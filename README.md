<div align="center">
<h1>Comprehensive Robustness Analysis of LiDAR-based 3D Object Detection in Autonomous Driving [ECCV 2026]</h1>

[**Adwait Chandorkar**](https://scholar.google.com/citations?user=2pxCiuIAAAAJ&hl=en) · [**Kai Krink**](https://scholar.google.com/citations?user=FsKmP1wAAAAJ&hl=en) · [**Yerdana Maulenbay**](https://scholar.google.ca/citations?hl=en&user=Tk1FBBQAAAAJ) · [**Hasan Tercan**](https://scholar.google.de/citations?user=QUzAdCUAAAAJ&hl=en) · [**Tobias Meisen**](https://scholar.google.com/citations?hl=de&user=fSmbntoAAAAJ&hl=en)

<a href="https://arxiv.org/abs/2607.02074"><img src='https://img.shields.io/badge/arXiv-2607.02074-red?logo=arXiv' alt='arXiv'></a>
<a href="https://tmdt-buw.github.io/adv-robustness-analy-3d-od/"><img src='https://img.shields.io/badge/Project-Robustness Analysis-green' alt='Project'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-Robustness Analysis-blue' alt='Paper BibTex'></a>

</div>

## 🎉 News 🎉
- 18.08.2026 : Added experiments for FGSM and PGD attacks on all the 3D-OD models.

## Overview

This is an official implementation of the paper ''Comprehensive Robustness Analysis of LiDAR-based 3D Object Detection in Autonomous Driving''. The repository provides an adversarial attack pipeline built upon the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) v1.0.0rc framework. It inlcudes evaluation and visualization.

Currently, the following adversarial attack methods have been adapted to work with the pipeline:

* **[IoU-S Attack](https://github.com/haichen-ber/IoU-S-Attack)**
* **FGSM & PGD** (own implementation)
* **[LiDAttack](https://github.com/Cinderyl/LiDAttack.git)**
* **[Non E2E](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.70011)** (own implementation)


It was tested using these models:
 1. **[Centerpoint](https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0.dev0/configs/centerpoint/README.md)**
 2. **[PillarNeSt](https://github.com/WayneMao/PillarNeSt)**
 3. **[PointPillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/README.md)**
 3. **[FocalFormer3D](https://github.com/NVlabs/FocalFormer3D.git)**

---

### Results
The table below show the ASR for each model/attack combination.

### 🚗 nuScenes

| Model | LiDAttack | Non E2E | IoU-S (Attach) | IoU-S (Perturb) | IoU-S (Detach) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centerpoint** | 0.81% | 48.60% | 23.30% | 88.46% | 43.26% |
| **FocalFormer3D** | 5.77% | 59.30% | 50.33% | 97.86% | 68.30% |
| **PillarNeSt** | 0.60% | 35.22% | 50.29% | 53.15% | 45.27% |
| **PointPillars** | 0.95% | 49.32% | 75.70% | 40.83% | 38.20% |

#### 🚙 Waymo

| Model | LiDAttack | Non E2E | IoU-S (Attach) | IoU-S (Perturb) | IoU-S (Detach) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centerpoint** | 0.00% | 71.49% | 33.50% | 71.49% | 74.36% |
| **FocalFormer3D** | 2.44% | 93.25% | 25.28% | 93.25% | 33.12% |
| **PillarNeSt** | 3.02% | 43.57% | 39.13% | 43.57% | 23.73% |
| **PointPillars** | 2.65% | 19.17% | 65.05% | 19.17% | 33.00% |

<br>

---

- **Point Cloud BEV Depth**<br>
  <img src="docs/images/7417_bev_depth.png" alt="Visual of the Point Cloud" width="800" />

- **Comparison of Adversarial vs. Clean Point Cloud**<br>
  <img src="docs/images/7417_bev_comparison.png" alt="Visual Comparing the adversarial and clean Point Cloud" width="800" />

- **Scene Object View**<br>
  <img src="docs/images/7417_obj_0.png" alt="Visual of an Object from the scene" width="450" />

These results and visuals were generated using the evaluation code in *data_processing/*.

## Installation
Most information can be found in the docs/ folder

#### If you are able to run and install MMDetection3d v1.0.0rc, mmcv-full v1.4.0 or older, continue with instructions below. If you have to run MMCV v2.x.x and MMDetection3d v1.1.0 or newer AND if you are not allowed to use conda/mamba environments, follow guides in

[docs/VOXELIZATION_MMCV2.md](VOXELIZATION_MMCV2.md) \
[docs/VOXELIZATION_MMCV2_USAGE.md](VOXELIZATION_MMCV2_USAGE) \
[docs/SETTING_UP_PILLARNEST_MMDET14.md](SETTING_UP_PILLARNEST_MMDET14) \
[docs/SETTING_UP_FOCALFORMER3D_MMDET14.md](SETTING_UP_FOCALFORMER3D_MMDET14) \
[docs/GRADIENT_EXTRACTION.md](GRADIENT_EXTRACTION) \
[mmdet3d_v1.4_files/slurm/README.md](mmdet3d_v1.4_files/slurm/README.md)

### Prerequisite: MMDetection3D v1.0.0rc

Before installing this attack pipeline, you need to have MMDetection3D v1.0.0rc installed and properly configured. Please refer to the detailed [MMDetection3D Installation Guide (docs/INSTALL_MMD3D.md)](docs/INSTALL_MMD3D.md) for step-by-step instructions. If you plan to use all models or want to make sure that everything works, it is recommended to replace/add all files listed in the mmdetection3d folder with the corresponding files in your mmdetection3d installation. A list of the required changes and more information regarding the installation can be found in the [MMDetection3D Installation Guide](docs/INSTALL_MMD3D.md).

**Important:** Some files in mmdetection3d must be changed to use this pipeline!

### Installing and Integrating the Adversarial Attack Pipeline

1. Clone this repository

2. Install additional Python dependencies:

```bash
pip install chamferdist
pip install mmengine
pip install numba==0.53
pip install open3d
pip install h5py
pip install imgaug
```

3. Adversarial Attack pipeline changes:
**`adversarial-attacks/adversarial_attack_pipeline.py`**:

Update the path variables to fit your Project! The only Prefix that needs to be changed is `PATH_PREFIX`, everything else is optional and based on preference. The variables can be found in lines 49-53:

```python
# Path variables
PATH_PREFIX = Path("/path/to/Projects")
SAVE_PATH = str(PATH_PREFIX / r"adversarial-attacks/visualizations")
orig_filename = "orig_results"
adv_filename = "adv_results"
```

### Setting up models
Refer to the docs on how to set up a model:
1. [Centerpoint](docs/SETTING_UP_CENTERPOINT.md)
2. [PillarNeSt](docs/SETTING_UP_PILLARNEST.md)
3. [PointPillars](docs/SETTING_UP_POINTPILLARS.md)
4. [FocalFormer3d](docs/SETTING_UP_FOCALFORMER.md)
5. [Other Models](docs/SETTING_UP_MODEL_TUTORIAL.md)

## Usage
The folder "scripts/" contains the bash/slurm scripts used for running the code. I recommend taking a look at these!
### Running an Adversarial Attack
Ths pipeline was designed to allow for independent slurm jobs to work simultaniously using the database, sidestepping a spconv bug that prevents us from using more than two gpus.

There are two ways to perform attacks. The first one is by providing a valid config and model weights, the second one is to use the presets (double check that the paths in the preset are accurate!). This will result in a database file that contains the relevant information.

**Hint:** Before using the presets, you need to change a few lines in the code! Change the base path to mmdetection3d on  line #53 to your mmdetection3d installation. Then check the presets paths and change them if needed (if your config name is different, or you save your weights in a different folder)

Example command to run an IoU-S detachment attack on a single gpu using a config and model weights:
```python
python adversarial_attack_pipeline.py --config path/to/model/config --model path/to/model/parameters --attack "iou_detachment" --num-samples 150 --num-drop 1024 --k-drop-round 16
```

Example command to run an IoU-S perturbation attack on a single gpu using a preset:
```python
python adversarial_attack_pipeline.py --preset-model "Pointpillars-waymo" --attack "iou_perturbation" --attack_lr 0.01 --steps 500
```
When using presets, the input config and model path are not needed as argument and will be ignored if given. Presets can also be used in the multi-gpu setting.

Example command to run an IoU-S detachment attack on a multi-gpu:
```python
torchrun --nproc_per_node=Num_GPUS adversarial_attack_pipeline.py \
    --config path/to/model/config --model path/to/model/parameters \
    --attack "iou_detachment" --num-samples 150 --num-drop 1024 --k-drop-round 16\
    --visualize --launcher pytorch \
    --checkpoint /optional/path/to/checkpoint.pkl
```
To sidestep the spconv bug, simply run multiple instances with an increasing `--base-rank` flag and refer them to the same save path, e.g for three gpus run all of these jobs:
```python
python adversarial_attack_pipeline.py --preset-model "Pointpillars-waymo" --attack "iou_perturbation" --attack_lr 0.01 --steps 500 --save-dir "results/" --base-rank 0
```
```python
python adversarial_attack_pipeline.py --preset-model "Pointpillars-waymo" --attack "iou_perturbation" --attack_lr 0.01 --steps 500 --save-dir "results/" --base-rank 1
```
```python
python adversarial_attack_pipeline.py --preset-model "Pointpillars-waymo" --attack "iou_perturbation" --attack_lr 0.01 --steps 500 --save-dir "results/" --base-rank 2
```

### Running Evaluation: 

To run the evaluation:
  ```python
  python data_processing/evaluation.py --data_path path/to/adversarial_results.db
  ```

For more CLI-Args options look in the table in [docs/CLI_ARGS.md](docs/CLI_ARGS.md).

**Output**:
The code will produce a sqlite3.db file containing two types of tables: *samples* and *boxes*. As the name already suggests, samples is a table containing computed information on a sample level, while boxes contains information for each box (the exact fields can be found in [sample.py](sample.py), line \#174 and line \#209 for the sample and box table respectively).

### Runnning Visualization:
To simply run the visualization:
  ```python
python data_processing/visualize.py \
    --db-path /path/to/adversarial_attack.db --samples id1 id2 id3 ... idn
  ```
Depending on what you want to visualize, look at the CLI Table in [docs/CLI_ARGS.md](docs/CLI_ARGS.md).

# Citation
```
@misc{chandorkar2026comprehensiverobustnessanalysislidarbased,
      title={Comprehensive Robustness Analysis of LiDAR-based 3D Object Detection in Autonomous Driving}, 
      author={Adwait Chandorkar and Kai Krink and Yerdana Maulenbay and Hasan Tercan and Tobias Meisen},
      year={2026},
      eprint={2607.02074},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.02074}, 
}
```
