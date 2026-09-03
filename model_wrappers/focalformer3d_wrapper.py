import torch
import torch.nn.functional as F
from model_wrappers.model_wrapper import ModelWrapper


class FocalFormer3DWrapper(ModelWrapper):
    """Wrapper for FocalFormer3D (Lidar-only and multi-modal).

    In v1.4, the plugin's pts_bbox_head.forward() is already differentiable
    and the config sets nms_type=None, so predict_by_feat() skips NMS.
    The old 470-line forward_decoder reimplementation is no longer needed.

    The only custom work is voxelization with gradient tracking (bypassing
    data_preprocessor's @torch.no_grad voxelize).
    """

    def __init__(self, model):
        super().__init__(model)

    def forward(self, **kwargs):
        """Differentiable forward through the full FocalFormer3D pipeline."""
        points = kwargs['points'][0][0]
        data_samples = kwargs.get('data_samples', [])

        pts_feats = self._extract_pts_feat([points])

        batch_input_metas = [ds.metainfo for ds in data_samples]
        outs = self.model.pts_bbox_head(
            [pts_feats, None], None, batch_input_metas)
        results = self.model.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas)

        return [{'pts_bbox': {
            'boxes_3d':  results[0].bboxes_3d,
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
        voxels.requires_grad_(True)

        voxel_features = self.model.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.model.pts_backbone(x)
        if self.model.with_pts_neck:
            x = self.model.pts_neck(x)
        return x
