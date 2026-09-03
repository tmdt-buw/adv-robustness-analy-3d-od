import torch
import torch.nn.functional as F
from model_wrappers.model_wrapper import ModelWrapper


class CenterPointWrapper(ModelWrapper):
    """Wrapper for CenterPoint and PillarNeSt.

    Provides a differentiable forward path by calling the encoder
    sub-modules directly (bypassing data_preprocessor's @torch.no_grad
    voxelize).
    """

    def __init__(self, model):
        super().__init__(model)

    def forward(self, **kwargs):
        """Differentiable forward. Returns old-format result list."""
        return self._forward_differentiable(**kwargs)

    # ------ differentiable forward path -----------------------------------

    def _forward_differentiable(self, **kwargs):
        points = kwargs['points'][0][0]          # (N, C)
        data_samples = kwargs.get('data_samples', [])

        pts_feats = self._extract_pts_feat([points])

        # Use the head's predict (applies bbox decoding, optional NMS)
        results = self.model.pts_bbox_head.predict(
            pts_feats, data_samples)

        return [{'pts_bbox': {
            'boxes_3d': results[0].bboxes_3d,
            'scores_3d': results[0].scores_3d,
            'labels_3d': results[0].labels_3d,
        }}]

    def _extract_pts_feat(self, pts_list):
        """Voxelize and encode point clouds WITH gradient tracking."""
        if not self.model.with_pts_bbox:
            return None

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

        # Keep gradients flowing through voxel features
        voxels.requires_grad_(True)

        voxel_features = self.model.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.model.pts_backbone(x)
        if self.model.with_pts_neck:
            x = self.model.pts_neck(x)
        return [x]
