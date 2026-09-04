import torch
import torch.nn.functional as F
from model_wrappers.model_wrapper import ModelWrapper


class PointPillarWrapper(ModelWrapper):
    """Wrapper for PointPillars (MVXFasterRCNN detector).

    Provides a differentiable forward path. Uses pts_voxel_encoder /
    pts_middle_encoder (MVXTwoStageDetector attribute names).
    """

    def __init__(self, model):
        super().__init__(model)

    def forward(self, **kwargs):
        """Differentiable forward. Returns old-format result list."""
        points = kwargs['points'][0][0]
        data_samples = kwargs.get('data_samples', [])

        pts_feats = self._extract_pts_feat([points])

        results = self.model.pts_bbox_head.predict(
            pts_feats, data_samples)

        return [{'pts_bbox': {
            'boxes_3d': results[0].bboxes_3d,
            'scores_3d': results[0].scores_3d,
            'labels_3d': results[0].labels_3d,
        }}]

    def _extract_pts_feat(self, pts_list):
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
        if not voxels.requires_grad:
            voxels.requires_grad_(True)

        voxel_features = self.model.pts_voxel_encoder(
            voxels, num_points, coors, None, None)
        batch_size = coors[-1, 0].item() + 1
        x = self.model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.model.pts_backbone(x)
        if self.model.with_pts_neck:
            x = self.model.pts_neck(x)
        return x
