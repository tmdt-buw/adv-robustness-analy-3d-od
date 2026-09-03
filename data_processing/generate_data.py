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

def prep_data(data, device='cuda:0'):
    '''
    Prepares Data for the model
    '''
    # store GTs separately
    # print("Data: ", data)
    data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0].data
    data['gt_labels_3d'] = data['gt_labels_3d'][0].data
    gt_bboxes_3d = data.pop('gt_bboxes_3d', None)
    gt_labels_3d = data.pop('gt_labels_3d', None)
    data['points'] = data['points'][0].data
    data['img_metas'] = data['img_metas'][0].data
    data = unwrap_data(data) 
    # Recursively move everything in the data dict to the correct device
    data = move_to_device(data, device)
    return data, gt_bboxes_3d, gt_labels_3d

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

    return dataset, data_loader, mm_model

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