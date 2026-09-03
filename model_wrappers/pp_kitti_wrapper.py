from model_wrappers.model_wrapper import ModelWrapper
import torch
from mmcv.parallel import DataContainer
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from pipeline_utils.utils import move_to_device
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
import math


class PPKittiWrapper(ModelWrapper):
    def __init__(self, model):
        super(PPKittiWrapper, self).__init__(model)
        self.dataset_name = "Kitti"


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

    def predict(self, return_loss=False,rescale=True, **kwargs) -> dict:
        """Run inference on a sample. Differs from the other models because the output format is slightly different"""
        with torch.no_grad():
            result = self.model(return_loss=False, **kwargs)
        return [{'pts_bbox': result[0]}]


    def forward(self, return_loss=True, **kwargs):
        """
        This function mimics the forward_test pass of mmdetection3d. The only difference is that the nms is cut out.
        """
        # Manually implement the forward pass to obtain the gradients etc. (Pointpillars NMS Head detaches them)
        results = self.forward_test(**kwargs)
        # Bring results in the correct format (same as the other models)
        return [{'pts_bbox': results[0]}]

# ------Model Forward Pass functions----------------------------------------------------

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
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.model.aug_test(points, img_metas, img, **kwargs)


    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        pts_feats = self.extract_feat(points, img_metas=img_metas)
        bbox_results = self.fake_head(pts_feats, img_metas, rescale=rescale)
        return bbox_results


    def fake_head(self, x, img_metas, score_thresh=0.05, rescale=False):
        """
        Differentiable head that returns decoded boxes, scores, labels WITHOUT NMS.

        Args:
            x (list[Tensor]): multi-level features from backbone/neck
            img_metas (list[dict])
            score_thresh (float): filter anchors by max class score
            rescale (bool): kept for API compatibility; not used

        Returns:
            list[dict]: length B, each dict has:
                'boxes_3d': LiDARInstance3DBoxes (N_i, 7) on the same device
                'scores_3d': Tensor (N_i,)
                'labels_3d': Tensor (N_i,)
        """
        # Forward through bbox head (multi-level lists)
        cls_scores, bbox_preds, dir_cls_preds = self.model.bbox_head(x)

        # Sanity
        assert isinstance(cls_scores, (list, tuple)) and isinstance(bbox_preds, (list, tuple))
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        B = cls_scores[0].shape[0]
        code_size = self.model.bbox_head.bbox_coder.code_size
        num_classes = self.model.bbox_head.num_classes
        num_anchors_per_loc = self.model.bbox_head.num_anchors

        # 1) Build per-level anchors with correct shapes, then expand to batch
        featmap_sizes = [feat.shape[-2:] for feat in cls_scores]  # list of (H, W)
        # grid_anchors returns a list[Tensor] of shape (H*W*num_anchors_per_loc, 7) per level
        mlvl_anchors = self.model.bbox_head.anchor_generator.grid_anchors(
            featmap_sizes, device=device
        )

        # 2) Reshape predictions per level to (B, N_level, *)
        mlvl_bbox = []
        mlvl_score = []
        mlvl_dir = [] if dir_cls_preds is not None else None

        for lvl in range(num_levels):
            H, W = featmap_sizes[lvl]
            # cls: [B, A*C, H, W] -> [B, H, W, A, C] -> [B, N, C]
            cls_lvl = cls_scores[lvl].permute(0, 2, 3, 1).contiguous()
            cls_lvl = cls_lvl.view(B, H, W, num_anchors_per_loc, num_classes)
            cls_lvl = cls_lvl.view(B, -1, num_classes)
            mlvl_score.append(cls_lvl)

            # bbox: [B, A*code, H, W] -> [B, H, W, A, code] -> [B, N, code]
            bbox_lvl = bbox_preds[lvl].permute(0, 2, 3, 1).contiguous()
            bbox_lvl = bbox_lvl.view(B, H, W, num_anchors_per_loc, code_size)
            bbox_lvl = bbox_lvl.view(B, -1, code_size)
            mlvl_bbox.append(bbox_lvl)

            if dir_cls_preds is not None:
                # dir: [B, A*2, H, W] -> [B, N, 2]
                dir_lvl = dir_cls_preds[lvl].permute(0, 2, 3, 1).contiguous()
                dir_lvl = dir_lvl.view(B, H, W, num_anchors_per_loc, 2)
                dir_lvl = dir_lvl.view(B, -1, 2)
                mlvl_dir.append(dir_lvl)

        # Concatenate across levels -> [B, N_all, *]
        bbox_pred = torch.cat(mlvl_bbox, dim=1)         # (B, N, code)
        cls_logit = torch.cat(mlvl_score, dim=1)        # (B, N, C)
        if dir_cls_preds is not None:
            dir_logit = torch.cat(mlvl_dir, dim=1)      # (B, N, 2)

        # Build matching anchors: list per level -> flatten each -> concat -> expand to batch
        flat_anchors = []
        for lvl_anchors in mlvl_anchors:
            # lvl_anchors: [1, num_z, H, W, num_rot, num_size, 7]
            flat_anchors.append(lvl_anchors.view(-1, lvl_anchors.size(-1)))  # (N_level, 7)

        anchors = torch.cat(flat_anchors, dim=0)           # (N_all, 7)
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)   # (B, N_all, 7)


        # 3) Decode boxes (keeps gradients)
        # DeltaXYZWLHRBBoxCoder expects anchors: (B, N, 7), deltas: (B, N, code)
        bbox_decoded = self.model.bbox_head.bbox_coder.decode(anchors, bbox_pred)  # (B, N, 7 or 9)
        bbox_decoded = bbox_decoded[..., :7]  # ensure (x, y, z, w, l, h, yaw)

        # Optional: normalize yaw into [-pi, pi] for stability; keeps grads
        # (You can copy the exact dir-bin correction from Anchor3DHead if needed)
        yaw = bbox_decoded[..., 6]
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        bbox_decoded = torch.cat([bbox_decoded[..., :6], yaw.unsqueeze(-1)], dim=-1)

        # 4) Scores & labels (sigmoid for PointPillars)
        scores = cls_logit.sigmoid()                     # (B, N, C)
        scores_max, labels = scores.max(dim=-1)          # (B, N), (B, N)

        # 5) Simple score filtering per batch (all differentiable masks)
        keep = scores_max > score_thresh                 # (B, N)

        results = []
        for b in range(B):
            kb = keep[b]
            boxes_b = bbox_decoded[b][kb]                # (Nb, 7)
            scores_b = scores_max[b][kb]                 # (Nb,)
            labels_b = labels[b][kb]                     # (Nb,)

            # Make LiDARInstance3DBoxes without breaking the graph
            # (LiDARInstance3DBoxes wraps a Tensor; it won’t detach)
            if boxes_b.numel() == 0:
                # Return empty containers on the correct device
                boxes3d = LiDARInstance3DBoxes(
                    boxes_b.new_zeros((0, 7)), origin=(0.5, 0.5, 0.5)
                )
                scores_b = scores_b.new_zeros((0,))
                labels_b = labels_b.new_zeros((0,), dtype=torch.long)
            else:
                boxes3d = LiDARInstance3DBoxes(boxes_b, origin=(0.5, 0.5, 0.5))

            results.append({
                'boxes_3d': boxes3d,
                'scores_3d': scores_b,
                'labels_3d': labels_b
            })

        return results



    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.model.voxelize(points)
        voxels.requires_grad_(True)
        voxel_features = self.model.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.model.middle_encoder(voxel_features, coors, batch_size)
        x = self.model.backbone(x)
        if self.model.with_neck:
            x = self.model.neck(x)
        return x

# ------ End of Model Forward Pass functions----------------------------------------------------