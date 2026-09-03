from model_wrappers.model_wrapper import ModelWrapper
import torch
import numpy as np
from mmcv.parallel import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from mmdet3d.core import (box3d_multiclass_nms, limit_period, xywhr2xyxyr)
from pipeline_utils.utils import move_to_device

class PointPillarWrapper(ModelWrapper):
    def __init__(self, model):
        super(PointPillarWrapper, self).__init__(model)


    def data_to_device(self, data, device):
        """
        Recursively move tensors to the device. This is only needed for the PointPillar forward imitation. 
        When using the utils.move_to_device it will throw an device mismatch error. When the utils.move_to_device 
        is replaced by this, the other models break.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self.data_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.data_to_device(x, device) for x in data]
        elif isinstance(data, tuple):
            return tuple(self.data_to_device(x, device) for x in data)
        elif isinstance(data, LiDARInstance3DBoxes):
            if data.tensor.device != device:
                return data.to(device)
            else:
                return data
        elif isinstance(data, DataContainer):
            return DataContainer(self.data_to_device(data.data, device))
        else:
            #print("Type not handled: ", type(data))
            return data

    def forward(self, return_loss=True, **kwargs):
        """
        This function mimics the forward_test pass of mmdetection3d. The only difference is that in simple_test_pts() the nms is cut out.
        """
        # Manually implement the forward pass to obtain the gradients etc. (Pointpillars NMS Head detaches them)
        return self.forward_test(**kwargs)
    
# ------Model Forward Pass functions----------------------------------------------------

    def forward_test(self, points, img_metas, img=None, **kwargs):
        return self.simple_test(points[0], img_metas[0], **kwargs)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        # Pointpillars does not use these!
        if img_feats and self.model.with_img_bbox:
            bbox_img = self.model.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def simple_test_pts(self, x, input_metas, rescale=False):
        """
        Modified simple_test_pts that returns decoded boxes, scores, labels
        in the same dict format as original, but WITHOUT NMS,
        so gradients can flow for adversarial attack.

        Args:
            x (list[Tensor]): features from backbone & neck
            input_metas (list[dict]): meta information of each sample.
            rescale (bool): whether to rescale boxes.

        Returns:
            list[dict]: bbox results in standard format.
        """
        # Forward through bbox head to get raw predictions
        cls_scores, bbox_preds, dir_cls_preds = self.model.pts_bbox_head(x)

        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # Taken from anchor3d_jead.py, execution of get_bboxes
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.model.pts_bbox_head.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.model.pts_bbox_head.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id] for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id] for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id] for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, None, rescale)
            result_list.append(proposals)

        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """ Taken from anchor3d_head.py
        Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.model.pts_bbox_head.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.model.pts_bbox_head.num_classes)
            if self.model.pts_bbox_head.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.model.pts_bbox_head.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.model.pts_bbox_head.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.model.pts_bbox_head.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.model.pts_bbox_head.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.model.pts_bbox_head.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        # HERE NMS HEAD HAPPENS, so we comment it out
        # results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
        #                                mlvl_scores, score_thr, cfg.max_num,
        #                                cfg, mlvl_dir_scores)
        # bboxes, scores, labels, dir_scores = results
        bboxes = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.model.pts_bbox_head.box_code_size)
        labels = mlvl_scores.argmax(dim=1)
        # if bboxes.shape[0] > 0:
        #     dir_rot = limit_period(bboxes[..., 6] - self.model.pts_bbox_head.dir_offset,
        #                            self.model.pts_bbox_head.dir_limit_offset, np.pi)
        #     bboxes[..., 6] = (
        #         dir_rot +  self.model.pts_bbox_head.dir_offset +
        #         np.pi * dir_scores.to(bboxes.dtype))
        # bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.model.pts_bbox_head.box_code_size)
        scores = mlvl_scores.max(dim=1)[0]
        return {
            'boxes_3d': bboxes,
            'scores_3d': scores,
            'labels_3d': labels
            }
        # return bboxes, scores, labels


        # # Generate anchors for decoding TODO: REMOVE OLD CODE
        # featmap_size = cls_scores[0].shape[-2:]  # (H, W)
        # mlvl_anchors = self.model.pts_bbox_head.anchor_generator.grid_anchors([featmap_size], device=device)[0]

        # # Apply sigmoid to classification scores (PointPillars uses sigmoid)
        # cls_score = cls_scores[0][0]  # [num_classes, H, W]
        # scores = cls_score.sigmoid().permute(1, 2, 0).reshape(-1, self.model.pts_bbox_head.num_classes)

        # # Decode bbox predictions
        # bbox_pred = bbox_preds[0][0].permute(1, 2, 0).reshape(-1, self.model.pts_bbox_head.bbox_coder.code_size)
        # bbox_decoded = self.model.pts_bbox_head.bbox_coder.decode(mlvl_anchors, bbox_pred)
        # bbox_decoded = bbox_decoded[:, :7]
        # # Get labels and max scores per box
        # labels = scores.argmax(dim=1)
        # scores_max = scores.max(dim=1)[0]

        # # To reduce the time taken during loss calculation, filter out some boxes that would probably be filtered out during nms
        # score_thresh = 0.05  # TODO: find good threshold

        # keep_mask = scores_max > score_thresh
        # bbox_decoded_filtered = bbox_decoded[keep_mask]
        # scores_filtered = scores_max[keep_mask]
        # labels_filtered = labels[keep_mask]

        # # Move filtered boxes to device if not already
        # bbox_decoded_filtered = self.data_to_device(bbox_decoded_filtered, self.device)

        # # Create LiDARInstance3DBoxes for decoded boxes
        # boxes_3d = LiDARInstance3DBoxes(bbox_decoded_filtered)

        # # TODO: Rescale boxes if needed (Update: Might not be needed, results look good without!)
        # #if rescale:
        #     # Note: Implement rescaling if you have scaling info in img_metas

        # return [{
        #         'boxes_3d': boxes_3d,
        #         'scores_3d': scores_max,
        #         'labels_3d': labels
        #         }]


    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.model.extract_img_feat(img, img_metas) # can ignore because PointPillars does not use these
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (None, pts_feats)


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.model.voxelize(pts)
        voxels.requires_grad_(True)
        voxel_features = self.model.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.model.pts_backbone(x)
        if self.model.with_pts_neck:
            x = self.model.pts_neck(x)
        return x

# ------End of Model Forward Pass functions----------------------------------------------------
