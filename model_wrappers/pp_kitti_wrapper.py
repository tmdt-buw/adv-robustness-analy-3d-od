import torch
import torch.nn.functional as F
import math
from model_wrappers.model_wrapper import ModelWrapper
from mmdet3d.structures import LiDARInstance3DBoxes


class PPKittiWrapper(ModelWrapper):
    """Wrapper for KITTI PointPillars (VoxelNet detector).

    Uses voxel_encoder / middle_encoder (VoxelNet attribute names).
    The differentiable forward skips NMS to keep gradients flowing.
    """

    def __init__(self, model):
        super().__init__(model)
        self.dataset_name = "Kitti"

    def forward(self, **kwargs):
        """Differentiable forward without NMS."""
        points = kwargs['points'][0][0]
        data_samples = kwargs.get('data_samples', [])

        pts_feats = self._extract_feat([points])
        results = self._fake_head(pts_feats, data_samples)
        return [{'pts_bbox': results[0]}]

    def _fake_head(self, x, data_samples, score_thresh=0.05):
        """Differentiable head: decoded boxes + scores WITHOUT NMS."""
        cls_scores, bbox_preds, dir_cls_preds = self.model.bbox_head(x)

        assert isinstance(cls_scores, (list, tuple))
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        B = cls_scores[0].shape[0]
        code_size = self.model.bbox_head.bbox_coder.code_size
        num_classes = self.model.bbox_head.num_classes
        num_anchors_per_loc = self.model.bbox_head.num_anchors

        featmap_sizes = [feat.shape[-2:] for feat in cls_scores]
        prior_gen = getattr(self.model.bbox_head, 'prior_generator',
                            getattr(self.model.bbox_head, 'anchor_generator', None))
        mlvl_anchors = prior_gen.grid_anchors(featmap_sizes, device=device)

        mlvl_bbox, mlvl_score = [], []
        for lvl in range(num_levels):
            H, W = featmap_sizes[lvl]
            cls_lvl = cls_scores[lvl].permute(0, 2, 3, 1).contiguous()
            cls_lvl = cls_lvl.view(B, H, W, num_anchors_per_loc, num_classes)
            cls_lvl = cls_lvl.view(B, -1, num_classes)
            mlvl_score.append(cls_lvl)

            bbox_lvl = bbox_preds[lvl].permute(0, 2, 3, 1).contiguous()
            bbox_lvl = bbox_lvl.view(B, H, W, num_anchors_per_loc, code_size)
            bbox_lvl = bbox_lvl.view(B, -1, code_size)
            mlvl_bbox.append(bbox_lvl)

        bbox_pred = torch.cat(mlvl_bbox, dim=1)
        cls_logit = torch.cat(mlvl_score, dim=1)

        flat_anchors = [a.view(-1, a.size(-1)) for a in mlvl_anchors]
        anchors = torch.cat(flat_anchors, dim=0).unsqueeze(0).expand(B, -1, -1)

        bbox_decoded = self.model.bbox_head.bbox_coder.decode(anchors, bbox_pred)
        bbox_decoded = bbox_decoded[..., :7]

        yaw = bbox_decoded[..., 6]
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        bbox_decoded = torch.cat([bbox_decoded[..., :6], yaw.unsqueeze(-1)], dim=-1)

        scores = cls_logit.sigmoid()
        scores_max, labels = scores.max(dim=-1)
        keep = scores_max > score_thresh

        results = []
        for b in range(B):
            kb = keep[b]
            boxes_b  = bbox_decoded[b][kb]
            scores_b = scores_max[b][kb]
            labels_b = labels[b][kb]

            if boxes_b.numel() == 0:
                boxes3d  = LiDARInstance3DBoxes(boxes_b.new_zeros((0, 7)), origin=(0.5, 0.5, 0.5))
                scores_b = scores_b.new_zeros((0,))
                labels_b = labels_b.new_zeros((0,), dtype=torch.long)
            else:
                boxes3d = LiDARInstance3DBoxes(boxes_b, origin=(0.5, 0.5, 0.5))

            results.append({
                'boxes_3d':  boxes3d,
                'scores_3d': scores_b,
                'labels_3d': labels_b,
            })
        return results

    def _extract_feat(self, pts_list):
        """Voxelize + encode with gradient tracking."""
        voxel_layer = self.model.data_preprocessor.voxel_layer
        voxels_list, coors_list, npoints_list = [], [], []
        for i, pts in enumerate(pts_list):
            v, c, n = voxel_layer(pts)
            coors_list.append(F.pad(c, (1, 0), mode='constant', value=i))
            voxels_list.append(v)
            npoints_list.append(n)

        voxels = torch.cat(voxels_list, dim=0)
        coors  = torch.cat(coors_list, dim=0)
        num_points = torch.cat(npoints_list, dim=0)
        voxels.requires_grad_(True)

        voxel_features = self.model.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.model.middle_encoder(voxel_features, coors, batch_size)
        x = self.model.backbone(x)
        if self.model.with_neck:
            x = self.model.neck(x)
        return x
