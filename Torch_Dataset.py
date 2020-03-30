import torch
from torch.utils import data
import torchvision.transforms.functional as TF


class Torch_Dataset(data.Dataset):
    '''
    Create Torch Dataset ready for Data Loaders.

    data_list: List of tuples from Dataset class.

    label_class: What class the labels are. 'on_off', 'dyskinesia', or 'tremor'.

    '''
    def __init__(self, data_list, label_class):
        self.data_list = data_list
        self.label_class = label_class
        self.label_2_index()

    def label_2_index(self):
        if self.label_class == 'on_off':
            self.label_index = 3
        elif self.label_class == 'dyskinesia':
            self.label_index = 4
        elif self.label_class == 'tremor':
            self.label_index = 5    

    def torch_transform(self, spec, label):
        spec = TF.to_tensor(spec)
        label = torch.tensor(label, dtype=torch.long)
        return spec, label
        
    def __getitem__(self, index):
        measurement = list(self.data_list[index])
        label = self.data_list[index][self.label_index]
        measurement[0], label = self.torch_transform(measurement[0], label)
        return measurement, label
    
    def __len__(self):
        return len(self.data_list)