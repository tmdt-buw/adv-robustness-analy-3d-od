import torch
import numpy as np
import copy
import math
from torch.utils.data import DataLoader
from mmcv.parallel import DataContainer
from mmcv import Config
from mmdet3d.core.bbox import BaseInstance3DBoxes
import torch.optim as optim
from chamferdist import ChamferDistance
from pipeline_utils.utils import move_to_device
dist_func = ChamferDistance()

import gc
import time

class ious_attack():
    def __init__(self, num_drop=1024, k_drop_round=16, sub_loss='all', attack_lr=0.01, steps=500, num_add=1024, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], voxel_size=[0.1, 0.1, 0.2], verbose = 2):
        # They drop a total of 1024 points with 16 points each iteration in their paper
        self.num_drop = num_drop
        self.k_drop_round = k_drop_round
        self.sub_loss = sub_loss
        # Default parameters for perturbation are lr=0.01 and steps=500
        self.attack_lr = attack_lr
        self.steps = steps
        # For Attachment lr=0.01 and steps=200, (num_add is not mentioned, in their code it defaults to 1024)
        self.num_add = num_add
        # For Voxel differentiation
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.verbose = verbose

    def set_pc_range(self, pc_range):
        self.point_cloud_range = pc_range


    def set_voxel_size(self, voxel_size):
        self.voxel_size = voxel_size


    def iou_detachment(self, data, model, gt_bboxes3d, gt_labels, device):
        """
        IoU-S Detachment Attack on point cloud using MMDetection3D model.
        Paper: https://ieeexplore.ieee.org/document/10557456
        This function is based on their implementation: https://github.com/haichen-ber/IoU-S-Attack
        Args:
            model: 3D object detection model
            data: Data from the dataset
            gt_bboxes3d: Ground truth boxes
            device: torch device
        
        Returns:
            result: Result of inference with adversarial point cloud
            torch.Tensor: Adversarial point cloud after T detachments
        """
        # Step 1: Getting the number of rounds
        num_rounds = math.ceil(self.num_drop / self.k_drop_round)
        # input point cloud
        #print(data)
        adv_point = data['points'][0][0].to(device)
        # Step 2: Iterate the following for the number of rounds
        for i in range(num_rounds):
            input_points = adv_point.clone().detach().requires_grad_(True)
            pc_shape = input_points.shape

            # number of points to drop in this round
            # test = self.num_drop - i * self.k_drop_round
            k_round = min(self.k_drop_round, self.num_drop - i * self.k_drop_round) #2
            # Step 3 + 4: Calculate adversarial loss and saliency map
            data['points'][0][0] = input_points
            result = model.forward(return_loss=False, rescale=True, **data)
            pre_bbox = result[0]['pts_bbox']['boxes_3d']
            pre_bbox = model.data_to_device(pre_bbox, device)
            pre_score = result[0]['pts_bbox']['scores_3d']

            gt_bbox = copy.deepcopy(gt_bboxes3d)
            gt_bbox = model.data_to_device(gt_bbox, device)
            ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox[0][0], pre_bbox)
            ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[1, 200]
            if ious_3d_sorted.shape[1]==0:
                adv_point = adv_point
            else:
                # adversarial loss
                loss = 0
                for j in range(idx.shape[-1]):
                    pre_score_truth = pre_score[idx[:,j]] #[9]
                    ious_3d = ious_3d_sorted[:, j] #[9]   
                    if self.sub_loss=='iou':
                        loss_ = -(torch.log(1-ious_3d+1e-8))
                    elif self.sub_loss=='score':
                        loss_ = -(torch.log(1 - pre_score_truth+1e-8))
                    elif self.sub_loss=='all':
                        loss_ = -(torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d))
                    loss = loss + loss_
                loss_all = loss.sum()
                loss_all.backward()
                #print(loss_all)
                # saliency map
                grad = input_points.grad.data
                # grad = model.grad(input_points, self.point_cloud_range, self.voxel_size, device)  #  [K, 5]
                #print("Grad data: ", grad)
                grad = torch.sum(grad ** 2, dim=1)  # [K]

                # Step 5: Detach k most salient points
                _, idx = (-grad).topk(k=pc_shape[0]-k_round, dim=-1) #[num]
                adv_point = input_points[idx]   

        # run inference on the adversarial point cloud
        data['points'][0][0] = adv_point
        result = model.predict(return_loss=False, rescale=True, **data)
        return result, adv_point


    def iou_perturbation(self, data, model, gt_bboxes3d, gt_labels, device):
        """
        IoU-S Perturbation Attack on point cloud using MMDetection3D model.
        Paper: https://ieeexplore.ieee.org/document/10557456
        This function is based on their implementation: https://github.com/haichen-ber/IoU-S-Attack
        Args:
            model: 3D object detection model
            data: Data from the dataset
            gt_bboxes3d: Ground truth boxes
            device: torch device
        Returns:
            result: Result of inference with adversarial point cloud
            adv_pointcloud: Adversarial point cloud after perturbations
        """
        torch.cuda.empty_cache()
        points = data['points'][0][0].to(device)
        #random start (1e-1 seems like a good start, lower performs worse and higher is visible)
        delta = torch.rand_like(points) * 1e-2
        adv_point = points + delta #only xyz
        # 点云不用clamp, 但5维点云的后2维不能改
        model_type = type(model).__name__
        if model_type=='CenterPoint':
            adv_point[:,-2:] = points[:,-2:]
        else:
            adv_point[:,-1:] = points[:,-1:]

        adv_point = adv_point.clone().detach().requires_grad_(True)
        opt = optim.Adam([adv_point], lr=self.attack_lr, weight_decay=0.)
        
        # Initialize best metrics
        # o_bestdist = 1e10
        # o_bestscore = 1e10
        # o_bestattack = None
        o_bestdist = np.array([1e10])
        o_bestscore = np.array([1e10])
        o_bestattack = np.zeros((1, adv_point.shape[0], adv_point.shape[1]))
        for step in range(self.steps):  

            data['points'][0][0] = adv_point.float()                 
            result = model.forward(return_loss=False, rescale=True, **data)
            pre_bbox = result[0]['pts_bbox']['boxes_3d']
            pre_score = result[0]['pts_bbox']['scores_3d']
            
            gt_bbox = copy.deepcopy(gt_bboxes3d)
            gt_bbox = model.data_to_device(gt_bbox, device)
            ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox[0][0], pre_bbox)
            ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[9, 200]
            
            if ious_3d_sorted.shape[1]==0:
                adv_point = adv_point
            else:
            
                if idx.shape[1]==0:
                    o_bestattack[0] = adv_point.detach().clone()
                else:
                    loss = 0
                    for j in range(idx.shape[-1]):
                        pre_score_truth = pre_score[idx[:,j]]#[9]
                        ious_3d = ious_3d_sorted[:, j] #[9]   
                        if self.sub_loss=='iou':
                            loss_ = -(torch.log(1-ious_3d+1e-8))
                        elif self.sub_loss=='score':
                            loss_ = -(torch.log(1 - pre_score_truth+1e-8))
                        elif self.sub_loss=='all':
                            loss_ = -(torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d+1e-8))
                        loss += loss_
                    #compute dist loss
                    dist1 = dist_func(adv_point[:, :3][None,:,:], points[:, :3][None,:,:])
                    dist2 = dist_func(points[:, :3][None,:,:], adv_point[:, :3][None,:,:])
                    dist_loss = dist1 + dist2
                    
                    dist_loss = dist_loss + torch.sqrt(torch.sum((adv_point[:, :3] - points[:, :3]) ** 2, dim=[0, 1]) + 1e-4)
                    loss_all = loss.sum() + dist_loss
                    opt.zero_grad()
                    # Add gradient of points to distance gradient
                    # grad = model.grad(adv_point.detach().clone(), self.point_cloud_range, self.voxel_size, device)
                    # def add_manual_grad(g):
                    #     if g is not None:
                    #         g_dist = g / (g.norm() + 1e-8)
                    #         g_adv = grad / (grad.norm() + 1e-8)
                    #         return (0.2*g_dist) + (2*g_adv)
                    #     else:
                    #         print("No initial grad!!!", flush=True)
                    #         return grad

                    # hook_handle = adv_point.register_hook(add_manual_grad)
                    # grad = adv_point.grad.data
                    # adv_point.grad.data += grad
                    loss_all.backward()
                    opt.step() 
                    # hook_handle.remove()
                    model_type = type(model).__name__
                    if model_type=='CenterPoint':
                        adv_point[:,-2:].data = points[:,-2:].data
                    else:
                        adv_point[:,-1:].data = points[:,-1:].data
                    if self.verbose > 2 and (step%100==0 or step==self.steps-1):
                        print('iteration {}, adv_loss: {:.4f}, dist_loss: {:.10f}'.format(step, loss.sum(), dist_loss.item()))
                        print('Loss all: ', loss_all.item())
                        # if grad_prev is not None:
                        #     cos_sim = torch.nn.functional.cosine_similarity(
                        #         grad_prev.clone().detach().flatten(), grad.clone().detach().flatten(), dim=0
                        #     ).item()
                        #     print("Gradient cosine similarity: ", cos_sim)
                        #     grad_diff = (grad.clone().detach() - grad_prev.clone().detach()).norm().item()
                        #     print('Gradient change: ', grad_diff)
                        # grad_prev = grad.clone().detach()
                    # record values!
                    # dist_val = torch.sqrt(torch.sum((adv_point[:, :3] - points[:, :3]) ** 2)).item()
                    dist_val = torch.sqrt(torch.sum((adv_point[:, :3] - points[:, :3]) ** 2, dim=[0, 1]))[None].detach().cpu().numpy()
                    if step==0:
                        dist_val = dist_val + 10.0
                    # adv_loss_val = loss_all.item()
                    adv_loss_val = loss_all[None].cpu().detach().numpy()
                    input_val = adv_point[None,:,:].detach().cpu().numpy()  # [K, 3]
                    # Update best
                    # if dist_val < o_bestdist and adv_loss_val < o_bestscore:
                    #     o_bestdist = dist_val
                    #     o_bestscore = adv_loss_val
                    #     o_bestattack = adv_point.detach().clone()
                    # update
                    for e, (dist, loss_1, ii) in enumerate(zip(dist_val, adv_loss_val, input_val)):
                        if dist < o_bestdist[e] and loss_1 < o_bestscore[e]:
                            o_bestdist[e] = dist
                            o_bestscore[e] = loss_1
                            o_bestattack[e] = ii   
                    # # for cuda memory
                    # adv_point.detach().requires_grad_()
        del adv_point, opt, loss_all, dist_loss, loss
        gc.collect()
        torch.cuda.empty_cache()
        final_pc = torch.from_numpy(o_bestattack[0]) 
        data['points'][0][0] = move_to_device(final_pc.float(), device)
        # run inference on the adversarial point cloud
        result = model.predict(return_loss=False, rescale=True, **data)
        return result, final_pc.float()


    def iou_attachment(self, data, model, gt_bboxes3d, gt_labels, device):
        """
        IoU-S Attachment Attack on point cloud using MMDetection3D model.
        Paper: https://ieeexplore.ieee.org/document/10557456
        This function is based on their implementation: https://github.com/haichen-ber/IoU-S-Attack
        Args:
            model: 3D object detection model 
            data: Data from the dataset
            gt_bboxes3d: Ground truth boxes
            device: torch device
        
        Returns:
            result: Result of inference with adversarial point cloud
            adv_point: Adversarial point cloud after T detachments
        """
        points = data['points'][0][0].to(device)
        #get critical point, points.shape (k, 5)
        input_pc = points.clone().detach().requires_grad_()
        data['points'][0][0] = input_pc
        result = model.forward(return_loss=False, rescale=True, **data)
        pre_bbox = result[0]['pts_bbox']['boxes_3d']
        pre_score = result[0]['pts_bbox']['scores_3d']
        
        gt_bbox = copy.deepcopy(gt_bboxes3d)
        gt_bbox = model.data_to_device(gt_bbox, device)
        ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox[0][0], pre_bbox)
        ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[9, 200]
        loss = 0
        for j in range(idx.shape[-1]):
            pre_score_truth = pre_score[idx[:,j]] #[9]
            ious_3d = ious_3d_sorted[:, j] #[9]   
            if self.sub_loss=='iou':
                loss_ = (torch.log(1-ious_3d+1e-8))
            elif self.sub_loss=='score':
                loss_ = (torch.log(1 - pre_score_truth+1e-8))
            elif self.sub_loss=='all':
                loss_ = (torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d))
            loss = loss + loss_
        loss_all = loss.sum()
        loss_all.backward()
        with torch.no_grad():
            # Inject gradient of points
            # grad = model.grad(input_pc, self.point_cloud_range, self.voxel_size, device)
            grad = input_pc.grad.data  #  [K, 5]
            grad = torch.sum(grad ** 2, dim=1)  #[K]
            _, idx = grad.topk(k=self.num_add, dim=-1)
            critical_points = points[idx]
        
        #init critical points with random start
        delta = torch.randn_like(critical_points) * 1e-7
        adv_point = critical_points + delta
        # 点云不用clamp, 但5维点云的后1维不能改
        model_type = type(model).__name__
        if model_type=='CenterPoint' or model_type=='PillarNest':
            adv_point[:,-2:] = critical_points[:,-2:]
        else:
            adv_point[:,-1:] = critical_points[:,-1:]
        #add attack
        adv_point = adv_point.clone().detach().requires_grad_()
        opt = optim.Adam([adv_point], lr=self.attack_lr, weight_decay=0.)
        
        for step in range(self.steps):  
            cat_data = torch.cat([points, adv_point], dim=0)
            # cat_data = cat_data.clone().detach().requires_grad_()
            data['points'][0][0] = cat_data
            result = model.forward(return_loss=False, rescale=True, **data)
            loss = 0
            pre_bbox = result[0]['pts_bbox']['boxes_3d']
            pre_score = result[0]['pts_bbox']['scores_3d']
            
            gt_bbox = copy.deepcopy(gt_bboxes3d)
            gt_bbox = model.data_to_device(gt_bbox, device)
            ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox[0][0], pre_bbox)
            ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1) #[9, 200]
            
            if ious_3d_sorted.shape[1]==0:
                adv_point = adv_point
            else:
            
                if idx.shape[1]==0:
                    adv_point = adv_point.clone().detach() 
                else:
                    loss = 0
                    for j in range(idx.shape[-1]):
                        pre_score_truth = pre_score[idx[:,j]] #[9]
                        ious_3d = ious_3d_sorted[:, j] #[9]   
                        if self.sub_loss=='iou':
                            loss_ = -(torch.log(1-ious_3d+1e-8))
                        elif self.sub_loss=='score':
                            loss_ = -(torch.log(1 - pre_score_truth+1e-8))
                        elif self.sub_loss=='all':
                            loss_ = -(torch.log(1-pre_score_truth+1e-8) + torch.log(1 - ious_3d+1e-8))
                        loss += loss_
                    #compute dist loss
                    dist1 = dist_func(adv_point[:, :3][None,:,:], points[:, :3][None,:,:])
                    # dist2 = dist_func(points_ori[:, :3][None,:,:], adv_point[:, :3][None,:,:])
                    dist_loss = dist1
                    loss_all = dist_loss + loss.sum()
                    opt.zero_grad()
                    loss_all.backward()
                    # # Inject gradient of points
                    # grad = model.grad(adv_point, self.point_cloud_range, self.voxel_size, device)
                    # adv_point.grad.data = grad
                    opt.step() 
                    model_type = type(model).__name__
                    if model_type=='CenterPoint' or model_type=='PillarNest':
                        adv_point[:,-2:].data = critical_points[:,-2:].data
                    else:
                        adv_point[:,-1:].data = critical_points[:,-1:].data
                    adv_point.detach().requires_grad_()
                    if self.verbose > 2 and (step%50==0 or step==self.steps-1):
                        print('iteration {}, adv_loss: {:.4f}, dist_loss: {:.10f}'.format(step, loss.sum(), dist_loss.item()))
        adv_point = torch.cat([points, adv_point], dim=0).float()
        data['points'][0][0] = move_to_device(adv_point, device)
        result = model.predict(return_loss=False, rescale=True, **data)
        return result, adv_point





