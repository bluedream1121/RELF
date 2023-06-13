import os, cv2, torch, yaml
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.dataset.correspondence_dataset import CorrespondenceDataset
from datasets.dataset.correspondence_database import CorrespondenceDatabase


def get_configs_from_yaml(cfg_file):
    with open(cfg_file, 'r') as f:
        overwrite_cfg = yaml.load(f,Loader=yaml.FullLoader)

    if 'default_config' in overwrite_cfg:
        with open(os.path.join(overwrite_cfg['default_config']), 'r') as f:
            default_train_config = yaml.load(f,Loader=yaml.FullLoader)
            config = overwrite_configs(default_train_config, overwrite_cfg)
    else:
        config = overwrite_cfg
    return config


class GIFTPerspectiveDataset(Dataset):
    def __init__(self, args):
            
        cfg_file = 'datasets/configs/GIFT-stage1.yaml' ## configs/GIFT-stage2.yaml

        config = get_configs_from_yaml(cfg_file)

        database = CorrespondenceDatabase()
        train_set = []
        for name in ['coco']:
            train_set += database.__getattr__(name + "_set")

        correspondence_dataset = CorrespondenceDataset(config, train_set)

        self.dataset = correspondence_dataset
        self.num_keypoints = args.num_keypoints
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
             
        return {'image1': data['image1'], 'pts1':data['pts1'][:self.num_keypoints], 'rotation': data['rotation'],
            'image2':data['image2'], 'pts2':data['pts2'][:self.num_keypoints], 'homography': data['homography']}
       