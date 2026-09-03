import pickle
import os
from os import path as osp
import argparse
from collections import defaultdict
import torch
from mmcv.parallel import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes


orig_filename = "orig_results"
adv_filename = "adv_results"

drop =  r"/path/to/Projects/adversarial-attacks/visualizations/run_iou_detachment_2025-07-17_10-50-31"
add =  r"/path/to/Projects/adversarial-attacks/visualizations/run_iou_attachment_2025-07-17_10-16-45"
pert =  r"/path/to/Projects/adversarial-attacks/visualizations/run_iou_perturbation_2025-07-17_10-56-24"

CLASS_ID_TO_NAME = {
}

def generate_class_name_dict(class_names):
    """
    Generates a dictionary from the class names given as input
    """
    global CLASS_ID_TO_NAME
    class_id_to_name = {i: class_name for i, class_name in enumerate(class_names)}
    class_id_to_name[-1] = 'Unknown'
    CLASS_ID_TO_NAME = class_id_to_name

    
def combine_checkpoints(path):
    """
    Combines the checkpoint file into one. 
    Args:
        path: path to the run files (parent file of checkpoint_{rank}.pkl)
    """
    
    all_sample_tokens = []
    all_per_sample = []
    all_per_class = []
    all_best_asr = []
    all_results = []
    i = 0
    while True:
        if not osp.isfile(osp.join(path,f"checkpoint_{i}.pkl")):
            break
        with open(osp.join(path,f"checkpoint_{i}.pkl"), 'rb') as f:
            checkpoint_data = pickle.load(f)
        results = []
        with open(osp.join(path,f"sample_results_{i}.pkl"), 'rb') as f:
            try:
                obj = pickle.load(f)
                # Case 1: old format — a single list at the start
                if isinstance(obj, list):
                    results.extend(obj)
                else:
                    # Mixed format: first element is not a list
                    results.append(obj)
                # Case 2: additional appended single items
                while True:
                    try:
                        results.append(pickle.load(f))
                    except EOFError:
                        break
            except EOFError:
                pass
        
        all_sample_tokens.append(checkpoint_data.get('sample_token_list', []))
        all_per_class.append(checkpoint_data.get('per_class', []))
        all_per_sample.append(checkpoint_data.get('per_sample', []))
        all_best_asr.append(checkpoint_data.get('best_asr_per_class', []))
        all_results.extend(results)
        i += 1

    full_sample_token_list = [token for sublist in all_sample_tokens for token in sublist]
    full_per_sample = [ps for plist in all_per_sample for ps in plist]
    full_per_class = [pc for pclist in all_per_class for pc in pclist]
    merged_sample_storage = {}
    merged_best_asr = {}
    # Go through all GPUs' best_asr_per_class dicts
    for gpu_idx, gpu_best_asr in enumerate(all_best_asr):
        for cls, info in gpu_best_asr.items():
            current_best = merged_best_asr.get(cls, {'asr': -1, 'sample_id': None})
            if info['asr'] > current_best['asr']:
                merged_best_asr[cls] = info
                # Copy the matching sample storage
                merged_sample_storage[info['sample_id']] = next((item for item in all_results if item["name"] == info['sample_id']), None)

    # Create final checkpoint
    checkpoint_data = {
        'save_path': path,
        'sample_token_list': full_sample_token_list,
        'per_sample': full_per_sample,
        'per_class': full_per_class,
        'sample_storage': merged_sample_storage,
        'best_asr_per_class': merged_best_asr
    }

    with open(osp.join(path, "checkpoint.pkl"), 'wb') as f:
        pickle.dump(checkpoint_data, f)

    with open(osp.join(path, "sample_results.pkl"), 'wb') as f:
        pickle.dump(all_results, f)

def write_results(attack, model, dataset, per_sample, per_class, save_path):
    separator1 = "=" * 100
    separator2 = "-" * 100
    file = osp.join(save_path, "results.txt")
    with open(file, "w") as f:
            f.write(f"{separator1}\n")
            f.write("Run Info:\n")
            f.write(f"{separator2}\n")
            f.write(f"Attack: {attack}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"{separator1}\n")
            f.write("Attack Results\n")
            f.write(f"{separator2}\n")
            f.write("[ Global Summary ]\n")
            f.write(per_sample.to_string(index=False, justify="center"))
            f.write("\n[ Per-Class Summary ]\n")
            f.write(per_class.to_string(index=False, justify="center"))
            f.write(f"\n{separator1}\n")


def visualize_sample(data, adv_pc, gt_bboxes_3d, adv_res, orig_res, save_path, out_dir, attack):
    """
    Visualizes the samples:
    1. The Original Sample with its ground truth and predicted boxes
    2. The Adversarial Sample with the gt and pred boxes
    3. The difference between the Original Sample and the Adversarial Sample
    """
    # Original Results
    show_results(data, gt_bboxes_3d, orig_res, save_path, orig_filename, verbose=3)
    # Adversarial Results
    data['points'][0][0] = adv_pc
    show_results(data, gt_bboxes_3d, adv_res, save_path, adv_filename, verbose=3)
    compare_pc(out_dir, orig_filename, adv_filename, mode=attack, verbose=3)

def unwrap_data(data):
    """Recursively unwrap DataContainers into their raw content."""
    if isinstance(data, DataContainer):
        return unwrap_data(data.data)
    elif isinstance(data, dict):
        return {k: unwrap_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [unwrap_data(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(unwrap_data(v) for v in data)
    else:
        return data

def move_to_device(data, device):
    """Recursively move tensors to the device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(x, device) for x in data)
    else:
        return data

    
def move_to_cpu_and_detach(obj):
    """
    Attempt to reduce CUDA memory used
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu_and_detach(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu_and_detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu_and_detach(v) for v in obj)
    else:
        return obj  # leave non-tensors as-is

        