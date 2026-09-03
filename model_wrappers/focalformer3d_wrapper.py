from model_wrappers.model_wrapper import ModelWrapper
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from mmdet.core import multi_apply

from mmdet3d.models import builder
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis

from mmdet3d.models.utils.utils import MLP, gen_sineembed_for_position, gen_sineembed_for_position_all
from mmdet3d.models.utils.transformer import *
from mmdet3d.models.utils.time_utils import T

import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import pdb

from mmdet3d.core.post_processing.merge_augs_ff import merge_aug_bboxes_3d_ff

class FocalFormer3DWrapper(ModelWrapper):
    """
    The Wrapper for FocalFormer3D (Lidar only and multi modal)
    """
    def __init__(self, model):
        super(FocalFormer3DWrapper, self).__init__(model)

        
    def forward(self, return_loss=False, **kwargs):
        """
        This function mimics the forward test path to get access to the gradients. The functionality is entirely the same!
        """
        return self.forward_test(**kwargs)
    
# ------Model Forward Pass functions-----------------------------------------------------------

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        img = [img] if img is None else img
        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        # with T('time', enable=True, sync=True, record=True):
        img_feats, pts_feats = self.model.extract_feat(
            points, img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas, rescale=rescale, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def simple_test_pts(self, x, x_img, img_metas, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """Test function of point cloud branch."""
        outs = self.forward_decoder(x, x_img, img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)
        if True:
            bbox_list = self.model.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        else:
            bbox_list = self.model.pts_bbox_head.get_heatmap_bboxes(
                outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    # ----- Decoder Head -----
    def forward_decoder(self, pts_inputs, img_inputs, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, **input_kwargs):
        self.model.pts_bbox_head.num_proposals = self.model.pts_bbox_head.num_proposals_ori # reset proposals
        
        lidar_feat = pts_inputs[0]
        if self.model.pts_bbox_head.extra_feat:
            extra_feats = pts_inputs[1][-1]
            pts_inputs[1].pop(-1)

        batch_size = lidar_feat.shape[0]
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.model.pts_bbox_head.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)
        if self.model.pts_bbox_head.multiscale:
            bev_pos_2 = self.model.pts_bbox_head.create_2D_grid(lidar_feat.shape[2] // 2, lidar_feat.shape[2] // 2).repeat(batch_size, 1, 1).to(lidar_feat.device) * 2
            bev_pos_4 = self.model.pts_bbox_head.create_2D_grid(lidar_feat.shape[2] // 4, lidar_feat.shape[2] // 4).repeat(batch_size, 1, 1).to(lidar_feat.device) * 4

        dense_heatmap_boxes = None
        query_box = None
        if not self.model.pts_bbox_head.multistage_heatmap:
            dense_heatmap = self.model.pts_bbox_head.heatmap_head(lidar_feat)
            if self.model.pts_bbox_head.input_img or self.model.pts_bbox_head.iterbev_wo_img:
                if isinstance(pts_inputs[1], (list, tuple)):
                    new_lidar_feat = pts_inputs[1][-1]
                else:
                    new_lidar_feat = pts_inputs[1]
                lidar_feat_flatten = new_lidar_feat.view(*lidar_feat_flatten.shape)

                dense_heatmap_img = self.model.pts_bbox_head.heatmap_head_img(new_lidar_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2
            else:
                heatmap = dense_heatmap.detach().sigmoid()
                new_lidar_feat = lidar_feat
            if self.model.pts_bbox_head.input_img or self.model.pts_bbox_head.iterbev_wo_img:
                heatmap_train = [dense_heatmap, dense_heatmap_img]
            else:
                heatmap_train = dense_heatmap
                    
            padding = self.model.pts_bbox_head.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.model.pts_bbox_head.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes':
                local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            elif self.model.pts_bbox_head.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # top #num_proposals among all classes
            top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.model.pts_bbox_head.num_proposals]
            top_proposals_class = top_proposals // heatmap.shape[-1]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            self.model.pts_bbox_head.query_labels = top_proposals_class

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.model.pts_bbox_head.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.model.pts_bbox_head.class_encoding(one_hot.float())
            query_feat = query_feat + query_cat_encoding

            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            query_heatmap_score = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.model.pts_bbox_head.num_classes, -1), dim=-1)
        else:
            dense_heatmap = self.model.pts_bbox_head.heatmap_head(lidar_feat) # original

            multistage_feats = pts_inputs[1]
            if self.model.pts_bbox_head.reuse_first_heatmap:
                multistage_feats.insert(0, lidar_feat)

            query_labels = []
            query_feats = []
            query_boxes = []
            query_poses = []
            query_heatmap_scores = []
            acc_masks = torch.ones_like(dense_heatmap).view(batch_size, -1)
            multistage_masks = []
            multistage_masks_independent_visualize = []
            heatmap_train = []
            multistage_bev_preds = []
            for i in range(self.model.pts_bbox_head.multistage_heatmap):
                if i == 0 and self.model.pts_bbox_head.reuse_first_heatmap:
                    if self.model.pts_bbox_head.heatmap_box:
                        assert self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.model.pts_bbox_head.thin_heatmap_box:
                            for task_id, task in enumerate(self.model.pts_bbox_head.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                if 'vel' in dense_pred:
                                    dense_pred = (dense_pred['reg'], dense_pred['height'], dense_pred['dim'], dense_pred['rot'], dense_pred['vel'])
                                else:
                                    dense_pred = (dense_pred['reg'], dense_pred['height'], dense_pred['dim'], dense_pred['rot'])
                                dense_pred = torch.cat(dense_pred, dim=1)[:, :, None].expand(-1, -1, self.model.pts_bbox_head.heatmap_tasks[task_id]['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                        else:
                            dense_heatmap_boxes_raw = self.model.pts_bbox_head.multi_stage_task_heads[i](shared_feat)
                            dense_preds_raw = torch.split(dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.model.pts_bbox_head.heatmap_tasks)):
                                dense_pred = torch.split(dense_preds_raw[task_id], [2, 1, 3, 2, 2], dim=1)
                                dense_preds.append( dict(reg=dense_pred[0], height=dense_pred[1], dim=dense_pred[2], rot=dense_pred[3], vel=dense_pred[4]) )
                                dense_heatmap_boxes.append( dense_preds_raw[task_id][:, :, None].expand(-1, -1,self.model.pts_bbox_head.heatmap_tasks[task_id]['num_class'], -1, -1) )
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(dense_heatmap_boxes, dim=2)

                    heatmap = dense_heatmap.detach().sigmoid()
                    heatmap_train.append(dense_heatmap)
                    multistage_masks.append(acc_masks.view(*heatmap.shape).clone())
                    heatmap = heatmap * acc_masks.view(*heatmap.shape) # remove early positive
                else:
                    if not self.model.pts_bbox_head.heatmap_box:
                        dense_heatmap_img = self.model.pts_bbox_head.heatmap_head_img[i]( multistage_feats[i] )
                    else:
                        assert self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes'
                        shared_feat = multistage_feats[i]
                        dense_preds = []
                        dense_heatmap_boxes = []
                        if not self.model.pts_bbox_head.thin_heatmap_box:
                            for task_id, task in enumerate(self.model.pts_bbox_head.multi_stage_task_heads[i]):
                                dense_preds.append(task(shared_feat))
                                dense_pred = dense_preds[-1]
                                dense_pred = torch.cat((dense_pred['reg'], dense_pred['height'], dense_pred['dim'], 
                                    dense_pred['rot'], dense_pred['vel']), dim=1)[:, :, None].expand(-1, -1, self.model.pts_bbox_head.heatmap_tasks[task_id]['num_class'], -1, -1)
                                dense_heatmap_boxes.append(dense_pred)
                            dense_heatmap_img = torch.cat([p['heatmap'] for p in dense_preds], dim=1)
                        else:
                            dense_heatmap_boxes_raw = self.model.pts_bbox_head.multi_stage_task_heads[i](shared_feat)
                            dense_preds_raw = torch.split(dense_heatmap_boxes_raw, [10] * 6, dim=1)
                            for task_id in range(len(self.model.pts_bbox_head.heatmap_tasks)):
                                dense_pred = torch.split(dense_preds_raw[task_id], [2, 1, 3, 2, 2], dim=1)
                                dense_preds.append( dict(reg=dense_pred[0], height=dense_pred[1], dim=dense_pred[2], rot=dense_pred[3], vel=dense_pred[4]) )
                                dense_heatmap_boxes.append( dense_preds_raw[task_id][:, :, None].expand(-1, -1, self.model.pts_bbox_head.heatmap_tasks[task_id]['num_class'], -1, -1) )
                            dense_heatmap_img = self.model.pts_bbox_head.heatmap_head_img[i]( multistage_feats[i] )
                        multistage_bev_preds.append(dense_preds)
                        dense_heatmap_boxes = torch.cat(dense_heatmap_boxes, dim=2)

                    heatmap = dense_heatmap_img.detach().sigmoid()
                    if i == 0:
                        heatmap_train.append(dense_heatmap)
                        multistage_masks.append(acc_masks.view(*heatmap.shape).clone())
                    heatmap = heatmap * acc_masks.view(*heatmap.shape) # remove early positive
                    heatmap_train.append(dense_heatmap_img)
                    multistage_masks.append(acc_masks.view(*heatmap.shape).clone())

                lidar_feat_flatten = multistage_feats[i].view(*lidar_feat_flatten.shape)

                padding = self.model.pts_bbox_head.nms_kernel_size // 2
                local_max = torch.zeros_like(heatmap)
                # equals to nms radius = voxel_size * out_size_factor * kenel_size
                local_max_inner = F.max_pool2d(heatmap, kernel_size=self.model.pts_bbox_head.nms_kernel_size, stride=1, padding=0)
                local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
                ## for Pedestrian & Traffic_cone in nuScenes
                if self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes':
                    local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                    local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
                elif self.model.pts_bbox_head.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                    local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                    local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
                heatmap = heatmap * (heatmap == local_max)
                heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

                # top #num_proposals among all classes
                top_proposals = torch.topk(heatmap.view(batch_size, -1), k=self.model.pts_bbox_head.num_proposals, dim=-1, largest=True, sorted=False).indices
                # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
                query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)

                query_labels.append(top_proposals_class)

                # add category embedding
                one_hot = F.one_hot(top_proposals_class, num_classes=self.model.pts_bbox_head.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.model.pts_bbox_head.class_encoding(one_hot.float())

                query_feat = query_feat + query_cat_encoding
                query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
                query_heatmap_score = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.model.pts_bbox_head.num_classes, -1), dim=-1)

                query_feats.append(query_feat)
                query_poses.append(query_pos)
                query_heatmap_scores.append(query_heatmap_score)

                if self.model.pts_bbox_head.heatmap_box:
                    box_dim = dense_heatmap_boxes.shape[1]
                    dense_heatmap_boxes = dense_heatmap_boxes.detach().view(batch_size, box_dim, self.model.pts_bbox_head.num_classes, heatmap.shape[-1])
                    assert self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes'
                    # learns from center_int to target offsets
                    dense_heatmap_boxes[:, :2, :, :] = dense_heatmap_boxes[:, :2, :, :] + bev_pos.int().float().transpose(1,2)[:, :, None].expand_as(dense_heatmap_boxes[:, :2, :, :])
                    dense_heatmap_boxes[:, 2:3, :, :] = dense_heatmap_boxes[:, 2:3, :, :].clip(min=-5., max=3.) # gravi center
                    dense_heatmap_boxes[:, 3:6, :, :] = dense_heatmap_boxes[:, 3:6, :, :].clip(min=np.log(0.5), max=np.log(15)) # box dim log
                    dense_heatmap_boxes[:, 6:8, :, :] = dense_heatmap_boxes[:, 6:8, :, :].clip(min=-1., max=1.) # sincos
                    dense_heatmap_boxes[:, 8:10, :, :] = dense_heatmap_boxes[:, 8:10, :, :].clip(min=-15., max=15.) # 
                    
                    dense_heatmap_boxes = dense_heatmap_boxes.view(batch_size, box_dim, self.model.pts_bbox_head.num_classes*heatmap.shape[-1])

                    query_box = dense_heatmap_boxes.gather(index=top_proposals[:, None, :].expand(-1, box_dim, -1), dim=-1)
                    query_boxes.append(query_box)

                ################ select to ignore ######################
                if self.model.pts_bbox_head.mask_heatmap_mode == 'pos':
                    selected_mask = acc_masks.new_zeros(batch_size, self.model.pts_bbox_head.num_classes, heatmap.shape[-1])
                    selected_mask.scatter(index=top_proposals_index[:, None, :].expand(-1, self.model.pts_bbox_head.num_classes, -1), dim=2, 
                        src=acc_masks.new_ones((batch_size, self.model.pts_bbox_head.num_classes, heatmap.shape[-1])))
                elif self.model.pts_bbox_head.mask_heatmap_mode == 'poscls':
                    selected_mask = acc_masks.new_zeros(batch_size, self.model.pts_bbox_head.num_classes * heatmap.shape[-1])
                    selected_mask.scatter(index=top_proposals, dim=1, src=torch.ones_like(top_proposals, dtype=acc_masks.dtype))
                elif self.model.pts_bbox_head.mask_heatmap_mode == 'boxcls':
                    boxmask_margin=1.
                    boxmask_margin_ratio=None
                    
                    assert self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes'
                    selected_mask = acc_masks.new_zeros(batch_size, self.model.pts_bbox_head.num_classes * heatmap.shape[-1])
                    selected_mask.scatter(index=top_proposals, dim=1, src=torch.ones_like(top_proposals, dtype=acc_masks.dtype))

                    def pos_inside_boxes(query_box, bev_pos, margin, min_bev_dim, margin_ratio=None): # bev_dim > 108 / 180 = 0.6
                        assert query_box.shape[1] >= 9
                        from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
                        rot, dim, center, height, vel = query_box[:, 6:8], query_box[:, 3:6], query_box[:, 0:2], query_box[:, 2:3], query_box[:, 8:]
                        query_boxes_std = self.model.pts_bbox_head.bbox_coder.decode_box(rot.clone(), dim.clone(), center.clone(), height.clone(), vel.clone())
                        pc_range = torch.as_tensor([-54, -54, -5.0, 54, 54, 3.0], device='cuda')
                        query_boxes_std[..., [0,]] = query_boxes_std[..., [0,]].clip(min=pc_range[0], max=pc_range[3])
                        query_boxes_std[..., [1,]] = query_boxes_std[..., [1,]].clip(min=pc_range[1], max=pc_range[4])
                        if margin_ratio is not None and margin_ratio > 0.:
                            query_boxes_std[..., [3,4]] *= (1. - margin_ratio)
                        else:
                            query_boxes_std[..., [3,4]] -= margin
                        query_boxes_std[..., [3,4]] = query_boxes_std[..., [3,4]].clip(min=min_bev_dim, max=10.)
                        query_boxes_std[..., 5] = 1000 # height -> max 
                        query_boxes_std[..., 2] = -100. # bottom center
                        temp_bev_pos = self.model.pts_bbox_head.bbox_coder.decode_center(bev_pos.transpose(1,2)) # bev points
                        temp_bev_pos = torch.cat([temp_bev_pos, temp_bev_pos.new_zeros(batch_size, 1, bev_pos.shape[1])], dim=1).transpose(1,2)
                        inside_boxes = points_in_boxes_gpu( 
                            temp_bev_pos, 
                            query_boxes_std[:, :, :7])

                        return inside_boxes

                    inside_boxes = pos_inside_boxes(query_box, bev_pos, margin=boxmask_margin, min_bev_dim=0.7, margin_ratio=boxmask_margin_ratio)
                    bev_pos_class = top_proposals_class.gather(index=inside_boxes.clip(min=0).long(), dim=1)
                    bev_pos_class[inside_boxes == -1] = self.model.pts_bbox_head.num_classes # background
                    selected_mask_box = acc_masks.new_zeros(batch_size, self.model.pts_bbox_head.num_classes + 1, heatmap.shape[-1])
                    selected_mask_box.scatter(index=bev_pos_class[:, None], dim=1, src=torch.ones_like(bev_pos_class[:, None], dtype=acc_masks.dtype))
                    selected_mask_box = selected_mask_box[:, :self.model.pts_bbox_head.num_classes].reshape(batch_size, self.model.pts_bbox_head.num_classes * heatmap.shape[-1])

                    selected_mask = (selected_mask + selected_mask_box > 0.1).float()
                else:
                    selected_mask = acc_masks.new_zeros(batch_size, self.model.pts_bbox_head.num_classes * heatmap.shape[-1])
                
                selected_mask = selected_mask.reshape(*dense_heatmap.shape)
                # masking by pooling
                selected_mask_kernel = F.max_pool2d(selected_mask, kernel_size=self.model.pts_bbox_head.nms_kernel_size, stride=1, padding=self.model.pts_bbox_head.nms_kernel_size // 2) 
                if self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes': ## for Pedestrian & Traffic_cone in nuScenes
                    selected_mask_kernel[:, 8:10] = F.max_pool2d(selected_mask[:, 8:10], kernel_size=1, stride=1, padding=0)
                elif self.model.pts_bbox_head.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                    selected_mask_kernel[:, 1:3] = F.max_pool2d(selected_mask[:, 1:3], kernel_size=1, stride=1, padding=0)
                
                acc_masks = acc_masks * (1.-selected_mask_kernel).view(*acc_masks.shape)
                
            self.model.pts_bbox_head.query_labels = torch.cat(query_labels, dim=1)
            query_feat = torch.cat(query_feats, dim=2)
            query_pos = torch.cat(query_poses, dim=1)
            query_heatmap_score = torch.cat(query_heatmap_scores, dim=2)
            if self.model.pts_bbox_head.heatmap_box:
                query_box = torch.cat(query_boxes, dim=2)

            self.model.pts_bbox_head.num_proposals = self.model.pts_bbox_head.num_proposals_ori * self.model.pts_bbox_head.multistage_heatmap

        if self.model.pts_bbox_head.training:
            self.model.pts_bbox_head.num_gts = [i.shape[0] for i in gt_labels_3d]
            self.model.pts_bbox_head.max_num_gts = max(self.model.pts_bbox_head.num_gts)

        query_labels = self.model.pts_bbox_head.query_labels
        if self.model.pts_bbox_head.training and self.model.pts_bbox_head.add_gt_groups > 0:
            updated_gt_queries = self.model.pts_bbox_head.generate_gt_groups(
                query_feat, query_pos, query_heatmap_score, 
                lidar_feat, lidar_feat_flatten, bev_pos, heatmap,
                gt_bboxes_3d, gt_labels_3d, dense_heatmap_boxes=dense_heatmap_boxes, query_box=query_box)
            if dense_heatmap_boxes is None:
                query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels = updated_gt_queries
            else:
                query_feat, query_pos, query_heatmap_score, batch_valid_gt_mask, batch_gt_query_labels, query_box = updated_gt_queries

            query_labels = torch.cat([query_labels, batch_gt_query_labels], dim=1)
                
        if self.model.pts_bbox_head.multiscale:
            if not self.model.pts_bbox_head.multistage_heatmap:
                lidar_feat = new_lidar_feat
            else:
                if self.model.pts_bbox_head.extra_feat:
                    lidar_feat = extra_feats
                else:
                    lidar_feat = multistage_feats[-1]

            multiscale_inputs = [lidar_feat]
            if self.model.pts_bbox_head.multiscale:
                multiscale_inputs.append( self.model.pts_bbox_head.dconv(multiscale_inputs[-1]) )
                multiscale_inputs.append( self.model.pts_bbox_head.dconv2(multiscale_inputs[-1]) )
            multiscale_inputs_flatten = torch.cat([i.flatten(2,3) for i in multiscale_inputs], dim=-1)

        ret_dicts = []
        for i in range(self.model.pts_bbox_head.num_decoder_layers):
            if self.model.pts_bbox_head.training:
                num_proposals_new = self.model.pts_bbox_head.num_proposals + self.model.pts_bbox_head.max_num_gts * self.model.pts_bbox_head.add_gt_groups
            else:
                num_proposals_new = self.model.pts_bbox_head.num_proposals
            
            prefix = 'last_' if (i == self.model.pts_bbox_head.num_decoder_layers - 1) else f'{i}head_'

            ################## Deformable Parameters #############
            if not self.model.pts_bbox_head.multiscale:
                W, H = lidar_feat.shape[-2:]
                spatial_shapes = torch.as_tensor([ [W, H] ], dtype=torch.long, device='cuda')
                level_start_index = torch.as_tensor([0,], dtype=torch.long, device='cuda')
            else:
                spatial_shapes = torch.as_tensor([ i.shape[2:] for i in multiscale_inputs ], dtype=torch.long, device='cuda')
                level_start_index = torch.as_tensor([0, *(torch.cumsum(torch.prod(spatial_shapes, dim=1), dim=0)[:-1])], dtype=torch.long, device='cuda')
                
                # lidar feat
                lidar_feat_flatten = multiscale_inputs_flatten
                if self.model.pts_bbox_head.bevpos and i == 0:
                    # bev pos
                    bev_pos = torch.cat([bev_pos, bev_pos_2, bev_pos_4], dim=1)

            if self.model.pts_bbox_head.training and self.model.pts_bbox_head.add_gt_groups > 0:
                # [batch_size, num_queries, num_keys]
                attn_masks = torch.ones((batch_size, num_proposals_new, num_proposals_new), dtype=bool, device='cuda')
                
                # all(query) sees query(key)
                attn_masks[:, :, :self.model.pts_bbox_head.num_proposals] = 0
                attn_masks[:, self.model.pts_bbox_head.num_proposals:, self.model.pts_bbox_head.num_proposals:] = torch.logical_not(batch_valid_gt_mask[:, None] & batch_valid_gt_mask[:, :, None])
                attn_masks = attn_masks[:, None, :, :].repeat(1, self.model.pts_bbox_head.num_heads, 1, 1).flatten(0,1)
            else:
                attn_masks = None
            
            kwargs = dict(
                spatial_shapes = spatial_shapes,
                level_start_index = level_start_index,
                valid_ratios = torch.ones((batch_size, 1, 2), device='cuda'),
                key_padding_mask = None,# key_padding_mask,
                attn_masks = attn_masks,
            )
            
            ################### Transformer Inputs ##############
            reference_points = query_pos / torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
            query_sine_pos = gen_sineembed_for_position(reference_points[:,:,:2])
            query_pos_embed = self.model.pts_bbox_head.pos_embed_learned[i](query_sine_pos) # bs, nq, 256
            if self.model.pts_bbox_head.boxpos is not None and query_box is not None:
                if self.model.pts_bbox_head.boxpos== 'xywlr': # actually boxdim3 + sincos2
                    extra_box_pos = query_box[:, 3:8].transpose(1,2)
                extra_box_sine_pos = gen_sineembed_for_position_all(extra_box_pos).flatten(-2)
                query_box_embed = self.model.pts_bbox_head.box_pos_embed_learned[i]
                query_pos_embed = query_pos_embed + query_box_embed

            # stage 1: query_pos = int+0.5, query_box = abs xy (z1)
            # stage 234..: query_pos = int+0.5, query_box = abs xy (z2)

            if self.model.pts_bbox_head.bevpos:
                bev_reference_points = bev_pos / torch.flip(spatial_shapes[:1], dims=(1,))[:, None]
                bev_sine_pos = gen_sineembed_for_position(bev_reference_points[:,:,:2])
                bev_pos_embed = self.model.pts_bbox_head.pos_embed_learned[i](bev_sine_pos) # bs, nq, 256
                pos_lidar_feat_flatten = lidar_feat_flatten + bev_pos_embed.transpose(1,2) #TODO( multiple addition for bev pos embedding )
            else:
                pos_lidar_feat_flatten = lidar_feat_flatten

            if self.model.pts_bbox_head.roi_feats and query_box is not None:
                rot, dim, center, height, vel = query_box[:, 6:8], query_box[:, 3:6], query_box[:, 0:2], query_box[:, 2:3], query_box[:, 8:]
                std_boxes = self.model.pts_bbox_head.bbox_coder.decode_box(rot.clone(), dim.clone() * self.model.pts_bbox_head.roi_expand_ratio[i], center.clone(), height.clone(), vel.clone())
                
                std_boxes = std_boxes.reshape(batch_size*num_proposals_new, std_boxes.shape[-1])
                lidar_std_boxes = LiDARInstance3DBoxes(std_boxes, box_dim=std_boxes.shape[-1]) # TODO(z is not checked)
                grid_points = self.model.pts_bbox_head.get_dense_grid_points(std_boxes, batch_size*num_proposals_new, self.model.pts_bbox_head.roi_feats)
                grid_points = torch.cat([grid_points, grid_points.new_ones(*grid_points.shape[:2], 1)], dim=-1)
                grid_points = rotation_3d_in_axis(grid_points, std_boxes[:, 6], axis=2)
                grid_points = grid_points[..., :2]
                grid_points = grid_points + std_boxes[:, None, :2]
                grid_points = grid_points.view(batch_size, num_proposals_new, self.model.pts_bbox_head.roi_feats**2, 2)
    
                if self.model.pts_bbox_head.test_cfg['dataset'] == 'nuScenes':
                    pc_range = torch.as_tensor([-54, -54, -5.0, 54, 54, 3.0], device='cuda')
                else:
                    pc_range = torch.as_tensor([-75.2, -75.2, -2, 75.2, 75.2, 4], device='cuda')
                grid_points = (grid_points - pc_range[:2]) / (pc_range[3:5] - pc_range[:2])
                grid_points = grid_points * 2. - 1.
                grid_points = grid_points.clip(min=-2., max=2.)

                if not self.model.pts_bbox_head.multiscale:
                    roi_feat = F.grid_sample(lidar_feat_flatten.view(batch_size, -1, *lidar_feat.shape[-2:]), grid_points) # W, H
                else:
                    ms_roi_feat = []
                    for feat in multiscale_inputs:
                        roi_feat = F.grid_sample(feat, grid_points, mode='bilinear')
                        ms_roi_feat.append(roi_feat)
                    roi_feat = torch.cat(ms_roi_feat, dim=1)
                roi_feat = roi_feat.permute(0, 2, 1, 3).reshape(batch_size * num_proposals_new, lidar_feat_flatten.shape[1] * (3 if self.model.pts_bbox_head.multiscale else 1) * self.model.pts_bbox_head.roi_feats**2)
                roi_feat = self.model.pts_bbox_head.roi_mlp(roi_feat)
                roi_feat = roi_feat.view(batch_size, num_proposals_new, lidar_feat_flatten.shape[1]).transpose(1,2)
                query_feat = query_feat + roi_feat

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            # query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            query_feat, reference_points = self.model.pts_bbox_head.decoder[i](
                query=query_feat.permute(2, 0, 1), 
                key=None,
                value=pos_lidar_feat_flatten.permute(2, 0, 1), 
                query_pos=query_pos_embed.permute(1, 0, 2), 
                reference_points=reference_points,
                **kwargs)

            query_feat = query_feat.permute(1, 2, 0)
            query_pos = reference_points * torch.flip(spatial_shapes[:1], dims=(1,))[:, None]

            # Prediction
            res_layer = self.model.pts_bbox_head.prediction_heads[i](query_feat)
            if self.model.pts_bbox_head.classaware_reg:
                for k in ['center', 'height', 'dim', 'rot']:
                    res_layer[k] = res_layer[k].view(batch_size, self.model.pts_bbox_head.num_classes, -1, num_proposals_new)
                    res_layer[k] = res_layer[k].gather(index=query_labels[:, None, None, :].expand(-1, -1, res_layer[k].shape[2], -1).clip(0, self.model.pts_bbox_head.num_classes - 1), dim=1)[:, 0]

            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

            if self.model.pts_bbox_head.roi_based_reg and query_box is not None: # only for bev
                res_layer['dim'][:,:2] = res_layer['dim'][:,:2] + query_box[:, 3:5].detach()
                res_layer['rot'] = res_layer['rot'] + query_box[:, 6:8].detach()
                # res_layer['vel'] = res_layer['vel'] + query_box[:, 8:].detach()
            
            query_box = [res_layer['center'], res_layer['height'], res_layer['dim'], res_layer['rot']]
            if 'vel' in res_layer:
                query_box.append(res_layer['vel'])
            query_box = torch.cat(query_box, dim=1).detach()
            ret_dicts.append(res_layer)

        if self.model.pts_bbox_head.training and self.model.pts_bbox_head.add_gt_groups > 0:
            ret_dicts[0]['batch_valid_gt_mask'] = batch_valid_gt_mask
            ret_dicts[0]['batch_gt_query_labels'] = batch_gt_query_labels
        if self.model.pts_bbox_head.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = query_heatmap_score  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = heatmap_train  #TODO(DeepInteraction only train image dense heatmap)
            if self.model.pts_bbox_head.multistage_heatmap:
                ret_dicts[0]['multistage_masks'] = multistage_masks

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key in ['dense_heatmap', 'query_heatmap_score', 'multistage_masks']:
                new_res[key] = ret_dicts[0][key]
            elif key in ['gt_query_bbox_targets']:
                new_res[key] = ret_dicts[0][key]
            elif key in ['batch_valid_gt_mask', 'batch_gt_query_labels']:
                new_res[key] = ret_dicts[0][key]
            else:
                if self.model.pts_bbox_head.training and self.model.pts_bbox_head.add_gt_groups > 0:
                    new_res[key] = torch.cat([
                        ret_dict[key][:, :, :(-self.model.pts_bbox_head.max_num_gts*self.model.pts_bbox_head.add_gt_groups)] 
                        for i, ret_dict in enumerate(ret_dicts)], dim=-1)
                    new_res[key + '_gtgroups'] = torch.cat([
                        ret_dict[key][:, :, -self.model.pts_bbox_head.max_num_gts*self.model.pts_bbox_head.add_gt_groups:] 
                        for i, ret_dict in enumerate(ret_dicts)], dim=-1)
                else:
                    new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
        if self.model.pts_bbox_head.heatmap_box:
            new_res['multistage_bev_preds'] = multistage_bev_preds
            new_res['query_pos'] = query_pos
            new_res['query_box'] = query_box
        return [[new_res]]
    

# ------End of Model Forward Pass functions----------------------------------------------------