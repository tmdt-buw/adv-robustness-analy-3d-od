from pipeline_utils.utils import unwrap_data, move_to_device
import torch

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.dataset_name = "NuScenes"

    def predict(self, return_loss=False,rescale=True, **kwargs) -> dict:
        """Run inference on a sample. Uses the original model inference"""
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **kwargs)
        return result

    def dataset_type(self):
        """Returns the name of the dataset as String"""
        return self.dataset_name

    def grad(self):
        print("Grad for that model_wrapper not implemented!")
        pass

    def set_device(self,device):
        self.device = device

    def prep_data(self, data, device='cuda:0'):
        '''
        Prepares Data for the model
        '''
        # print("Data keys: ", data.keys())
        # store GTs separately
        data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0].data
        data['gt_labels_3d'] = data['gt_labels_3d'][0].data
        gt_bboxes_3d = data.pop('gt_bboxes_3d', None)
        gt_labels_3d = data.pop('gt_labels_3d', None)
        # print("Data: ", data)
        data['points'] = data['points'][0].data
        data['img_metas'] = data['img_metas'][0].data
        data = unwrap_data(data) 
        # Recursively move everything in the data dict to the correct device
        data = move_to_device(data, device)
        return data, gt_bboxes_3d, gt_labels_3d

    def data_to_device(self, data, device):
        return move_to_device(data,device)
