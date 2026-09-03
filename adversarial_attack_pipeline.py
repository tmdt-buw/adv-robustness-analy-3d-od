import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['FILAMENT_DISABLE_LOGGING'] = '1' #to remove some warnings (didnt work)
from os import path as osp
import argparse
import mmcv
import torch
import numpy as np
import pandas as pd
import math
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataloader
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer
from mmcv.runner import init_dist, get_dist_info
from torch.utils.data.distributed import DistributedSampler
from mmcv.parallel import MMDistributedDataParallel
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
import torch.distributed as dist
import torch.optim as optim
from itertools import islice
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import pickle
import torch.multiprocessing as mp
import logging
import gc
import copy
import time

# Own imports
from attacks.ious_attack import ious_attack
from attacks.FGSM import FGSM
from attacks.LiDAttack import LiDAttack
from pipeline_utils.utils import combine_checkpoints, unwrap_data, move_to_device, move_to_cpu_and_detach, write_results, generate_class_name_dict
from model_wrappers.centerpoint_wrapper import CenterPointWrapper
from model_wrappers.pointpillar_wrapper import PointPillarWrapper
from model_wrappers.pp_kitti_wrapper import PPKittiWrapper
from model_wrappers.focalformer3d_wrapper import FocalFormer3DWrapper
from pipeline_utils.db_util import init_db, check_available, save_res, progress, check_any_taken_incomplete

# CUDA OOM Bug analysis tools
from GPUMemoryInspector import quick_gpu_diagnosis, GPUMemoryInspector, monitor_gpu_memory

# Path variables
PATH_PREFIX = Path("/path/to/project")
SAVE_PATH = str(PATH_PREFIX / r"ECCV2026/visualizations")
orig_filename = "orig_results"
adv_filename = "adv_results"

# From config (changed in main to config values later)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]


def main(config=None, model_path=None, reduced=False, attack=None, preset=None, save_path=None, checkpoint=None, db_name="adversarial_attack.db", device='cuda:0', launcher=None):
    # Distributed Launch variables
    distributed = False
    if launcher != None:
        init_dist(launcher)
        distributed = True
    rank, world_size = get_dist_info()

    if distributed:
        torch.cuda.set_device(rank)

    # Presets to make it easier. Custom can be used when different model weights/configs are wanted, but they need to be given through args
    model_name = "Custom_preset"
    dataset_name = "Custom_preset"
    # NuScenes Models (None selftrained)
    if preset == "centerpoint":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/centerpoint_attacks/centerpoint_nus_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/centerpoint_attacks/centerpoint_nus_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth")
        model_name = "Centerpoint"
        dataset_name = "NuScenes"

    elif preset == "pillarnest":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pillarnest/pillarnest_nus_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pillarnest/pillarnest_nus_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/pillarnest_base.pth")
        model_name = "Pillarnest"
        dataset_name = "NuScenes"

    elif preset == "pointpillars":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pointpillars/pointpillars_nus_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pointpillars/pointpillars_nus_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth")
        model_name = "Pointpillars"
        dataset_name = "NuScenes"

    elif preset == "focalformer3d":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_L_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_L_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/FocalFormer3D_nus.pth")
        if lc_fusion: #todo: add reduced for this
            if reduced:
                config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_LC_adv_red.py")
            else:
                config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_LC_adv.py")
            model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/FocalFormer3D_LC_nus.pth")
        model_name = "FocalFormer3D"
        dataset_name = "NuScenes"

    elif preset == "pointpillars-kitti":
        config = str(PATH_PREFIX / r"mmdetection3d/configs/pointpillars/pointpillars_kitti_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/pointpillars_selftrained.pth")
        model_name = "Pointpillars"
        dataset_name = "Kitti"

    elif preset == "centerpoint-kitti":
        config = str(PATH_PREFIX / r"mmdetection3d/configs/centerpoint_attacks/centerpoint_kitti_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/centerpoint_kitti.pth")
        model_name = "Centerpoint"
        dataset_name = "Kitti"

    elif preset == "pillarnest-kitti":
        config = str(PATH_PREFIX / r"mmdetection3d/configs/pillarnest/pillarnest_kitti_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/pillarnest_base_kitti.pth")
        model_name = "Pillarnest"
        dataset_name = "Kitti"

    elif preset == "pointpillars-waymo":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pointpillars/pointpillars_waymo_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pointpillars/pointpillars_waymo_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/pointpillars_waymo.pth")
        model_name = "Pointpillars"
        dataset_name = "Waymo"

    elif preset == "centerpoint-waymo":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/centerpoint_attacks/centerpoint_waymo_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/centerpoint_attacks/centerpoint_waymo_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/centerpoint_waymo.pth")
        model_name = "Centerpoint"
        dataset_name = "Waymo"

    elif preset == "pillarnest-waymo":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pillarnest/pillarnest_waymo_adv_red.py")
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/pillarnest/pillarnest_waymo_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/pillarnest_waymo.pth")
        model_name = "Pillarnest"
        dataset_name = "Waymo"

    elif preset == "focalformer3d-waymo":
        if reduced:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_Waymo_L_adv_red.py") #todo!
        else:
            config = str(PATH_PREFIX / r"mmdetection3d/configs/focalformer3d/FocalFormer3D_Waymo_L_adv.py")
        model_path = str(PATH_PREFIX / r"mmdetection3d/checkpoints/FocalFormer3D_Waymo.pth")
        model_name = "FocalFormer3D"
        dataset_name = "Waymo"
    # Other Models. WARNING: If other detector structures are used it is probably necessary to create a custom wrapper!
    elif preset == "custom":
        if config is None or model_path is None:
            raise RuntimeError("If no preset is chosen config and model_path cannot be None!")

    if save_path is None:
        save_path = SAVE_PATH

    # Memory inspector (Trying to figure out where the memory leaks)
    inspector = GPUMemoryInspector(rank=rank)
    if cmm and verbose >= 1:
        print("Initial GPU state:")
        inspector.print_gpu_tensor_report(top_n=10)

    if checkpoint:
        save_path = checkpoint
        if rank == 0:
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        if distributed:
            dist.barrier()
    else:
        # Create new sub folder for each new run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if debug:
            path = Path(save_path) /"debug"/ f"{model_name}" / f"{reduced_pre}{dataset_name}" / f"{attack}" / f"run_{file_prefix}{timestamp}"
        else:
            path = Path(save_path) / f"{model_name}" / f"{reduced_pre}{dataset_name}" / f"{attack}" / f"run_{file_prefix}{timestamp}"
        if rank == 0:
            path.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()
        save_path = str(path)
    if rank == 0 and verbose > 1:
        print_run_info(attack, preset, dataset_name, model_name, save_path)
    
    db_path = os.path.join(save_path, db_name)
    if rank == 0:
        init_db(save_path, db_path=db_path)
        if verbose > 0 and checkpoint: 
            print(f"[Rank {rank}] Loaded checkpoint with {progress(db_path)} completed samples.")

    if distributed: # Wait until db is ready!
        dist.barrier()
    
    cfg = Config.fromfile(config)
    # Set pc range and voxel size
    point_cloud_range = cfg.point_cloud_range
    voxel_size = cfg.voxel_size
    ious.set_pc_range(point_cloud_range)
    ious.set_voxel_size(voxel_size)
    generate_class_name_dict(cfg.class_names)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Build model and dataset
    dataset, data_loader, model = load_model_and_dataset(cfg, model_path, device=device, distributed=distributed)

    if num_samples:
        num_iterations = int(num_samples/world_size)
        if rank == 0 and verbose >= 2:
            print("Iterating through ", num_samples, " Samples!")
    else:
        state = "reduced" if reduced else "full"
        if rank == 0 and verbose >= 2:
            print(f"Iterating through the {state} dataset, ", len(data_loader)*world_size, " Samples!")
        num_iterations = len(data_loader)
        
    # if no num_samples is given iterate over entire dataloader
    data_iterator = islice(data_loader, num_iterations)
    #start = time.time()
    # Get one sample from the dataloader (Contains a datapoint with all information about the samples)
    for data_i, data_point in enumerate(data_iterator):
        # To counteract CUDA-OOM errors, makeshift solution
        try:
            main_iteration(model, model_name, attack, dataset_name, data_i, data_point, save_path, inspector, cmm, world_size, rank, verbose, db_path, device)
        except Exception as e:
            print(f"ERROR in sample {(data_i*world_size)+rank}: {e}", flush=True)
            if not debug:
                print(f"Rerunning Sample {(data_i*world_size)+rank}",flush=True)
                # print("GPU state at error:")
                if cmm:
                    inspector.print_gpu_tensor_report(top_n=10)
                    inspector.find_memory_leaks()
                # Give it another chance
                gc.collect()
                torch.cuda.empty_cache()
                main_iteration(model, model_name, attack, dataset_name, data_i, data_point, save_path, inspector, cmm, world_size, rank, verbose, db_path, device)
            else:
                raise e
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    #print("Loop took ", time.time()-start, "s")
    if rank == 0 and verbose >= 0:
        if num_samples:
            print(f"Compeleted first iteration: ", progress(db_path), "/", num_samples, "Samples completed!")
        else:
            print(f"Compeleted first iteration: ", progress(db_path), "/", len(data_loader)*world_size, "Samples completed!")
    # Case that samples were taken but not completed:
    while(check_any_taken_incomplete(db_path, attack, model_name, dataset_name, (base_rank*world_size)+rank)):
        for data_i, data_point in enumerate(data_iterator):
            main_iteration(model, model_name, attack, dataset_name, data_i, data_point, save_path, inspector, cmm, world_size, rank, verbose, db_path, device)

    if distributed: # Wait to prevent being stuck
            dist.barrier()
    if rank == 0 and verbose >= 0:
        if num_samples:
            print(f"Compeleted all claimed samples: ", progress(db_path), "/", num_samples)
        else:
            print(f"Compeleted all claimed samples: ", progress(db_path), "/", len(data_loader)*world_size)

    
    
def main_iteration(model, model_name, attack, dataset_name, data_i, data_point, save_path, inspector, cmm, world_size, rank, verbose, db_path, device):
    """
    Contains the logic of the main loop. This has been seperated from the loop hoping that it would solve the CUDA-OOM Error
    """
    # Allows to use more than 2 gpus by using multiple scripts
    worker_id = (base_rank*world_size)+rank
    # monitor_gpu_memory only inspects tensors and prints when cmm is True. Otherwise it does nothing
    with monitor_gpu_memory(inspector, f"Sample {(data_i*world_size)+rank} - Data Prep", cmm):
        data_point = move_to_device(data_point, device)
        sample_token = data_point['img_metas'][0].data[0][0]['sample_idx']

        available = check_available(db_path, attack, model_name, dataset_name, sample_token, worker_id)

        # If already attacked in checkpoint skip
        if not available:
            del data_point
            return
            
        # Prepare data
        data, gt_bboxes_3d, gt_labels_3d = model.prep_data(data_point, device)

    with monitor_gpu_memory(inspector, f"Sample {(data_i*world_size)+rank} - Original Prediction",cmm):
        # Run model to get original results
        orig_res = model.predict(return_loss=False, rescale=True, **data)

    with monitor_gpu_memory(inspector, f"Sample {(data_i*world_size)+rank} - Adversarial Attack",cmm):
        # Adversarial attack
        adv_res, adv_pc = adversarial_attack(copy.deepcopy(data), model, attack, gt_bboxes_3d, gt_labels_3d, device)
        save_res(db_path, save_path, orig_res, move_to_cpu_and_detach(data['points'][0][0]), adv_res, move_to_cpu_and_detach(adv_pc), attack, model_name, dataset_name, sample_token, gt_bboxes_3d, gt_labels_3d,worker_id)

    # Detailed CUDA Memory print 
    if cmm and (data_i + 1) % 5 == 0:
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS AFTER {data_i + 1} SAMPLES")
        print(f"{'='*60}")
        inspector.print_gpu_tensor_report(top_n=15)
        inspector.find_memory_leaks()
        inspector.print_memory_fragmentation()

    # Clear space to prevent CUDA out of memory errors
    cleanup_vars = ['data', 'gt_bboxes_3d', 'gt_labels_3d', 'orig_res', 'adv_res', 'adv_pc', "sample_data_i", "sample_data"]
    for var in cleanup_vars:
        if var in locals():
            del locals()[var]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def print_summary(ps, pc, data_i=None):
    separator = "=" * 100
    if data_i is None:
        print(f"\n{separator}\nBatch Results\n{separator}\n", flush=True)
    else:
        print(f"\n{separator}\nSample {data_i} Results\n{separator}\n", flush=True)

    print("[ Global Summary ]", flush=True)
    print(ps.to_string(index=False, justify='center'), flush=True)

    print("\n[ Per-Class Summary ]", flush=True)
    print(pc.to_string(index=False, justify='center'), flush=True)

    print(f"\n{separator}\n", flush=True)

def print_run_info(attack, preset, dataset, model, save_path):
    separator = "=" * 100
    print(f"\n{separator}\n", flush=True)
    print("Run Info:", flush=True)
    print(f"\n{separator}\n", flush=True)
    print(f"Attack: {attack}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    print(f"Saving data to: {save_path}")
    print(f"Base Rank: {base_rank}")
    print(f"\n{separator}\n", flush=True)

def visualize_sample(sample, out_dir, attack, dataset = "NuScenes"):
    # Original Results
    show_results(sample['points'], sample['gt_boxes'], sample['result'], out_dir, orig_filename, verbose=verbose)
    # Adversarial Results
    show_results(sample['adv_points'], sample['gt_boxes'], sample['adv_result'], out_dir, adv_filename, verbose=verbose)
    compare_pc(out_dir, sample['points'], sample['adv_points'], mode=attack, verbose=verbose)


def adversarial_attack(sample, model, attack, gt_bboxes3d, gt_labels_3d, device):
    """
    TODO! My Idea here is that we pass the input argument string here and simply have an IF statement to swap between the different attacks
    """
    if attack == 'iou_perturbation':
        result, adv_pc = ious.iou_perturbation(sample, model, gt_bboxes3d, gt_labels_3d, device)
    elif attack == 'iou_detachment':
        result, adv_pc = ious.iou_detachment(sample, model, gt_bboxes3d, gt_labels_3d, device)
    elif attack == 'iou_attachment':
        result, adv_pc = ious.iou_attachment(sample, model, gt_bboxes3d, gt_labels_3d, device)
    elif attack == 'fgsm':
        result, adv_pc = fgsm.fgsm_attack(sample, model, gt_bboxes3d, gt_labels_3d, device)   
    elif attack == 'pgd':
        result, adv_pc = fgsm.pgd_attack(sample, model, gt_bboxes3d, gt_labels_3d, device)   
    elif attack == 'lidattack':
        result, adv_pc = lidattack.attack(sample, model, gt_bboxes3d, gt_labels_3d, device)  
    else:
        print("WARNING! ", attack, " has not been implemented! Returning None")
        return None
    return result, adv_pc


def load_model_and_dataset(cfg, model_path, device='cuda:0', distributed=False):
    # Build dataset
    dataset = build_dataset(cfg.data.test)

    # Fix: Some custom datasets don't initialize `flag`, used by GroupSampler
    if not hasattr(dataset, 'flag'):
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)

    # Use DistributedSampler if distributed training is enabled
    # Build data loader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        shuffle=False,
        dist=distributed
    )

    # Initialize model
    if model_path:
        mm_model = init_model(cfg, checkpoint=model_path, device=device)
    else:
        mm_model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        mm_model.cfg = cfg
        mm_model = mm_model.to(device)

    mm_model.eval()
    # Wrap model in DDP if needed
    if distributed:
        mm_model = mm_model.cuda()
    else:
        mm_model = mm_model.to(device)

    # Wrapping in custom model wrapper to keep functions consistant across models. When adding new models, new wrappers are probably needed
    model_type = type(mm_model).__name__
    # The model type for PillarNest also returns "CenterPoint", this is no problem because everything for CenterPoint works for PillarNest. Centerpoint_f is a fixed version of centerpoint that works for Kitti
    if model_type =='CenterPoint' or model_type =='PillarNest'or model_type == 'CenterPoint_f':
        model = CenterPointWrapper(mm_model)
    # MVXFasterRcNN is used by PointPillars as detector
    elif model_type=='MVXFasterRCNN':
        model = PointPillarWrapper(mm_model)
    # Kitti PointPillars uses a different detection head and needs to be treated differently
    elif model_type=='VoxelNet':
        model = PPKittiWrapper(mm_model)
    # FocalFormer Wrapper
    elif model_type=='FocalFormer3D':
        model = FocalFormer3DWrapper(mm_model)
    else:
        print(f"WARNING: {model_type} not implemented! Defaulting to CenterPoint Wrapper")
        model = CenterPointWrapper(mm_model)

    model.set_device(device)

    return dataset, data_loader, model


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    model_preset_list = ["centerpoint", "pillarnest", "pointpillars", "focalformer3d",
                        "centerpoint-kitti", "pillarnest-kitti", "pointpillars-kitti", "focalformer3d-kitti",
                        "pointpillars-waymo", "centerpoint-waymo", "pillarnest-waymo", "focalformer3d-waymo",
                        "custom"]
    attack_list = ["iou_detachment", "iou_attachment", "iou_perturbation", "fgsm", "pgd", "lidattack"]
    parser = argparse.ArgumentParser(description='Adversarial Attack Pipeline for mmdetection3d')
    parser.add_argument('--preset-model',dest="preset_model", default="custom", choices=model_preset_list, type=str.lower, help='Model Preset Path+Config from Code')
    parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--reduced', default=False, action="store_true", help="Should the reduced version of the dataset be used instead of the whole dataset?")
    parser.add_argument('--lc-fusion', dest="lc_fusion", default=False, action="store_true", help="Use Fusion weights?")
    parser.add_argument('--model', default=None, help='Path to model checkpoints')
    parser.add_argument('--checkpoint', default=None, help='Path checkpoints, usually previous run directory')
    parser.add_argument('--num-samples', default=None, dest="num_samples", type= int, help='Amount of samples attacked')
    parser.add_argument('--attack', default='iou_detachment', choices=attack_list, help='Attack that will be used')
    parser.add_argument('--save-dir', dest="save_dir", default=None, help='Directory for saving visualizations')
    parser.add_argument('--no-visual', dest="no_visual", default=False, action='store_true', help='Skip visualizing Results?')
    parser.add_argument('--launcher', default=None, help='Multi-GPU attacks?')
    # arguments for IoU-S attacks
    parser.add_argument('--sub_loss', type= str, default='iou', choices= ['iou', 'score', 'all'])
    # Detachment attack
    parser.add_argument('--num-drop', dest="num_drop", type= int, default=1024, help='Total number of Points that will be dropped')
    parser.add_argument('--k-drop-round', dest="k_drop_round", type= int, default=16, help='Number of Points that will be dropped per round')
    # Perturbation attack (and Attachment parameters)
    parser.add_argument('--attack_lr', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=500, help='add and perturbation steps')
    # Attachment attack
    parser.add_argument('--num_add', type=int, default=1024, help='num points of add')
    # LidAttack arguments
    parser.add_argument('--gen_iterations', type=int, default=100, help='maximum number of iterations for genetic algorithm')
    parser.add_argument('--population', type=int, default=20, help='size of population')
    # FGSM and PGD arguments
    parser.add_argument('--epsilon', type=float, default=0.3, help='magnitude of FGSM/PGD perturbation')
    parser.add_argument('--iterations', type=int, default=1000, help='number of PGD iterations')
    parser.add_argument('--step-size', dest="step_size", type=float, help='size of PGD step per iteration')

    # QOL arguments
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode with set parameters?')
    parser.add_argument('--prefix', type= str, default='', help='Prefix for file name')
    parser.add_argument('--base-rank', dest="base_rank", type=int, default=0, help='base rank for global worker id assignment')

    # cmm is to monitor CUDA Memory. Especially when using IoUs_perturbation, CUDA OOM Errors occur frequently
    parser.add_argument('--cuda-memory-monitor', '--cmm', dest="cmm", default=False, action='store_true', help='Monitor CUDA Memory. Prints tensors and storage after every operation')
    parser.add_argument('--verbosity', type=int, default=2, help='Verbosity level: 0->Silent, 1->Important Outputs, 2->All Outputs, 3->Everything!')

    args = parser.parse_args()
    visualize = not args.no_visual
    lc_fusion = args.lc_fusion
    sub_loss = args.sub_loss
    num_samples = args.num_samples
    
    debug = args.debug
    base_rank = args.base_rank
    cmm = args.cmm
    verbose = args.verbosity

    # Faster computations for debug
    if debug:
        print("WARNING: DEBUG MODE ACTIVE!!! Your input parameters will be ignored!")
        ious = ious_attack(num_drop=1000, k_drop_round=100, attack_lr=0.01, steps=300, num_add=1000, verbose=3)
        fgsm = FGSM(epsilon = args.epsilon, step_size=args.step_size, iterations = 10)
        lidattack = LiDAttack(fitness_threshold=0.5, max_iterations=1000, population_size=5, mutation_rate=0.1, verbose=3)
        # num_samples = 16
        if args.verbosity == 2:
            verbose = 3
    else:
        ious = ious_attack(num_drop=args.num_drop, k_drop_round=args.k_drop_round, sub_loss='all', attack_lr=args.attack_lr, steps=args.steps, num_add=args.num_add, verbose=verbose)
        fgsm = FGSM(epsilon = args.epsilon, iterations = args.iterations, step_size=args.step_size) #TODO: add values from CLA
        lidattack = LiDAttack(fitness_threshold=0.5, max_iterations=args.gen_iterations, population_size=args.population, mutation_rate=0.1, verbose=verbose)

    if args.reduced:
        reduced_pre=f"reduced_"
    else:
        reduced_pre=''
    file_prefix=args.prefix

    if args.checkpoint:
        if verbose > 0:
            print("Resuming attacks from: ", args.checkpoint)

    # Has launcher when using multi gpu.
    if args.launcher:
        main(args.config, args.model, args.reduced, args.attack, preset=args.preset_model, save_path = args.save_dir, checkpoint=args.checkpoint, launcher=args.launcher)
    else:
        main(args.config, args.model, args.reduced, args.attack, preset=args.preset_model, save_path = args.save_dir, checkpoint=args.checkpoint)

    if debug:
        print("WARNING: DEBUG MODE ACTIVE!!! Your input parameters were ignored!")

