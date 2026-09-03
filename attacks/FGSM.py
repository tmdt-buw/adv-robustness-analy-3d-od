import torch
import copy
from pipeline_utils.utils import unwrap_data, move_to_device
from mmdet3d.core.bbox import BaseInstance3DBoxes



class FGSM():
    """
    Fast Gradient Sign Method and Iterative FGSM (PGD)
    Implementation based on this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8803770
    """
    def __init__(self, epsilon = 0.3, iterations = 1, step_size = None, verbose = 2):
        """
        Args:
            epsilon: attack strength (magnitude of perturbation)
            iterations: number of iterations for PGD
            step_size: the magnitude of each step in PGD. If set to None computes the step_size automatically
            verbose: show prints?
        """
        self.epsilon = epsilon
        self.iterations = iterations
        self.verbose = verbose
        self.step_size = step_size
        # Automatic step size computation, taken from: https://github.com/zzj403/BEV_Robust/blob/main/apis_common/test_pgd_imgpoint.py#L69
        if step_size is None:
            self.step_size = epsilon / (iterations-2) if iterations > 2 else epsilon/2

    def set_pc_range(self, pc_range):
        self.point_cloud_range = pc_range

    def fgsm_attack(self, data, model, gt_bboxes3d, gt_labels, device):
        """
        Basic implementation of FGSM for point clouds.
        Args:
            data: data used for the model
            model: model that is used
            gt_bboxes3d: ground truth boxes
            gt_labels: ground truth labels
            device: cuda/cpu device
        """
        # preparation for fgsm
        input_points = data['points'][0][0].to(device).clone().detach().requires_grad_(True)
        orig_points = copy.deepcopy(input_points).detach().requires_grad_(False)
        data['points'][0][0] = input_points

        result = model.forward(return_loss=False, rescale=True, **data)
        pre_bbox = result[0]['pts_bbox']['boxes_3d']
        pre_bbox = model.data_to_device(pre_bbox, device)
        pre_score = result[0]['pts_bbox']['scores_3d']

        gt_bbox = copy.deepcopy(gt_bboxes3d)
        gt_bbox = model.data_to_device(gt_bbox, device)
        # find matches between gt and pred
        ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox[0][0], pre_bbox)
        # if there are no iou overlaps we cant compute the loss, the base model fails already
        if ious_3d.shape[-1] > 0:
            _, idx = ious_3d.max(dim=-1)
            matched_scores = pre_score[idx]  # [N_gt]

            # loss computation (using cross entropy)
            loss = -torch.log(1 - matched_scores + 1e-8).sum()
            loss.backward()

            # FGSM update
            grad = input_points.grad.data
            adv_pc = input_points+self.epsilon*torch.sign(grad)

            # Only perturb xyz
            adv_pc[:, 3:] = orig_points[:, 3:]
            data['points'][0][0] = adv_pc
        else:
            adv_pc = input_points
    
        # Adv result
        adv_result = model.predict(return_loss=False, rescale=True, **data)

        return adv_result, adv_pc

    def pgd_attack(self, data, model, gt_bboxes3d, gt_labels, device):
        """
        Runs the Point Gradient Descent attack! If iterations is set to 1 and step_size = epsilon, it equals the FGSM attack!
        Args:
            data: data used for the model
            model: model that is used
            gt_bboxes3d: ground truth boxes
            gt_labels: ground truth labels
            device: cuda/cpu device
        """

        # Initialize adversarial points
        adv_points = data['points'][0][0].to(device).clone().detach().requires_grad_(True)
        # Original points
        orig_points = copy.deepcopy(adv_points).detach().requires_grad_(False)


        for _ in range(self.iterations):
            adv_points = adv_points.to(device).clone().detach().requires_grad_(True)
            data['points'][0][0] = adv_points
            # Forward
            result = model.forward(return_loss=False, rescale=True, **data)
            pre_bbox = model.data_to_device(
                result[0]['pts_bbox']['boxes_3d'], device
            )
            pre_score = result[0]['pts_bbox']['scores_3d']

            # Match using IoU
            with torch.no_grad():
                gt_bbox = model.data_to_device(
                    copy.deepcopy(gt_bboxes3d), device
                )
                ious_3d = BaseInstance3DBoxes.overlaps(
                    gt_bbox[0][0], pre_bbox
                )
            if ious_3d.shape[-1] > 0:
                _, idx = ious_3d.max(dim=-1)
                matched_scores = pre_score[idx]

                # Attack loss
                loss = -torch.log(1 - matched_scores + 1e-8).sum()
                # reset grads
                model.model.zero_grad()
                if adv_points.grad is not None:
                    adv_points.grad.zero_()
                loss.backward()

                with torch.no_grad():
                    adv_points[:, :3] += self.step_size * torch.sign(adv_points.grad[:, :3])

                    delta = adv_points[:, :3] - orig_points[:, :3]
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)

                    adv_points[:, :3] = orig_points[:, :3] + delta
                    adv_points[:, 3:] = orig_points[:, 3:]

                adv_points.requires_grad_(True)

        # Final prediction
        data['points'][0][0] = adv_points.detach()
        adv_result = model.predict(return_loss=False, rescale=True, **data)

        return adv_result, adv_points.detach()


    def custom_data_preprocess(self, data, input_points, device):
        data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0].data
        data['gt_labels_3d'] = data['gt_labels_3d'][0].data
        data['img_metas'] = data['img_metas'][0].data
        data = unwrap_data(data) 
        data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0][0]
        data['points'][0] = input_points
        print(type(data["gt_bboxes_3d"]))
        print(type(data["gt_bboxes_3d"][0]))

        print("GT Boxes", data["gt_bboxes_3d"])
        # Recursively move everything in the data dict to the correct device
        data = move_to_device(data, device)
        # data['gt_bboxes_3d'][0] = data['gt_bboxes_3d'][0].to(device)
        return data
