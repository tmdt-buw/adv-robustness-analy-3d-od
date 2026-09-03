from model_wrappers.model_wrapper import ModelWrapper

class CenterPointWrapper(ModelWrapper):
    """
    The Wrapper for both CenterPoint and PillarNeSt
    """
    def __init__(self, model):
        super(CenterPointWrapper, self).__init__(model)

        
    def forward(self, return_loss=False, **kwargs):
        """
        This function mimics the forward test path to get access to the gradients. The functionality is entirely the same!
        """
        #return self.model(return_loss=return_loss, **kwargs)
        return self.forward_test(**kwargs)

# ------Model Forward Pass functions----------------------------------------------------

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Function taken and adapted from mmdetection3d/mmdet3d/models/detectors/base.py 
        Mimics the forward pass during testing but gives access to the gradients.
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

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """
        Function taken and adapted from mmdetection3d/mmdet3d/models/detectors/mvx_two_stage.py 
        Mimics the forward pass during testing but gives access to the gradients.
        Test function without augmentaiton.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.model.with_pts_bbox:
            bbox_pts = self.model.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.model.with_img_bbox:
            bbox_img = self.model.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def extract_feat(self, points, img, img_metas):
        """
        Function taken and adapted from mmdetection3d/mmdet3d/models/detectors/mvx_two_stage.py 
        Mimics the forward pass during testing but gives access to the gradients.
        Extract features from images and points.
        """
        img_feats = self.model.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """
        Function taken and adapted from mmdetection3d/mmdet3d/models/detectors/centerpoint.py 
        Mimics the forward pass during testing but gives access to the gradients.
        Extract features of points.
        """
        if not self.model.with_pts_bbox:
            return None
        voxels, num_points, coors = self.model.voxelize(pts)
        voxels.requires_grad_(True)
        voxel_features = self.model.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.model.pts_backbone(x)
        if self.model.with_pts_neck:
            x = self.model.pts_neck(x)
        return x

# ------End of Model Forward Pass functions----------------------------------------------------
