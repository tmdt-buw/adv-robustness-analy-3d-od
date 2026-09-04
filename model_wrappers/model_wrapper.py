from pipeline_utils.utils import move_to_device
import torch


class ModelWrapper:
    """Base wrapper that translates between the adversarial-attack pipeline's
    internal data format and the mmdet3d v1.4 model API.

    Internal format (used by attack code):
        data = {
            'points': [[points_tensor]],   # data['points'][0][0] = (N, C)
            'data_samples': [Det3DDataSample],
        }
        gt_bboxes_3d, gt_labels_3d  # extracted separately

    v1.4 model API:
        model.test_step({'inputs': batch_inputs, 'data_samples': batch_data_samples})
    """

    def __init__(self, model):
        self.model = model
        self.dataset_name = "NuScenes"

    # ----- public interface used by the attack pipeline ------------------

    def predict(self, **kwargs) -> list:
        """Run inference (with NMS). Returns old-format result list."""
        batch_inputs, batch_data_samples = self._to_v14(kwargs)
        batch_data_samples = [ds.clone() for ds in batch_data_samples]
        with torch.no_grad():
            results = self.model.test_step({
                'inputs': batch_inputs,
                'data_samples': batch_data_samples
            })
        return self._from_v14(results)

    def dataset_type(self):
        return self.dataset_name

    def grad(self):
        print("Grad for that model_wrapper not implemented!")

    def set_device(self, device):
        self.device = device

    def prep_data(self, data_list, device='cuda:0'):
        """Convert v1.4 dataloader output to the pipeline's internal format.

        Args:
            data_list: dict or list of dicts from pseudo_collate (batch of 1)
            device: target device

        Returns:
            data - dict with 'points' and 'data_samples'
            gt_bboxes_3d - ground-truth boxes wrapped in [[...]]
            gt_labels_3d - ground-truth labels wrapped in [[...]]
        """
        if isinstance(data_list, dict):
            sample = data_list
        else:
            sample = data_list[0]

        ds = sample['data_samples']
        if isinstance(ds, (list, tuple)):
            ds = ds[0]

        gt_bboxes_3d = getattr(ds.gt_instances_3d, 'bboxes_3d', None) if hasattr(ds, 'gt_instances_3d') else None
        gt_labels_3d = getattr(ds.gt_instances_3d, 'labels_3d', None) if hasattr(ds, 'gt_instances_3d') else None

        if gt_bboxes_3d is not None and hasattr(gt_bboxes_3d, 'to'):
            gt_bboxes_3d = gt_bboxes_3d.to(device)
        if gt_labels_3d is not None and hasattr(gt_labels_3d, 'to'):
            gt_labels_3d = gt_labels_3d.to(device)

        points = sample['inputs']['points']
        if isinstance(points, (list, tuple)):
            points = points[0]
        points = move_to_device(points, device)

        data = {
            'points': [[points]],           # attack code uses data['points'][0][0]
            'data_samples': [ds.to(device) if hasattr(ds, 'to') else ds],
        }
        return data, [[gt_bboxes_3d]], [[gt_labels_3d]]

    def data_to_device(self, data, device):
        return move_to_device(data, device)

    # ----- internal helpers ----------------------------------------------

    @staticmethod
    def _to_v14(kwargs):
        """Translate internal format -> v1.4 model API."""
        points = kwargs['points'][0][0]
        data_samples = kwargs.get('data_samples', [])
        batch_inputs = {'points': [points]}
        return batch_inputs, data_samples

    @staticmethod
    def _from_v14(results):
        """Translate v1.4 Det3DDataSample results -> old dict format."""
        out = []
        for r in results:
            pred = r.pred_instances_3d
            out.append({'pts_bbox': {
                'boxes_3d': pred.bboxes_3d,
                'scores_3d': pred.scores_3d,
                'labels_3d': pred.labels_3d,
            }})
        return out
