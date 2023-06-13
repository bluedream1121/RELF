import os, torch, argparse
import numpy as np
from tqdm import tqdm

from .model.extraction_tools import initialize_networks, compute_kpts_desc

class KeyNet_extractor:
    def __init__(self, num_kpts = 5000):
        
        keynet_config = {

            'KeyNet_default_config':
                {
                    # Key.Net Model
                    'num_filters': 8,
                    'num_levels': 3,
                    'kernel_size': 5,

                    # Trained weights
                    'weights_detector': 'baselines/extract_KeyNet/model/weights/keynet_pytorch.pth',
                    'weights_descriptor': 'baselines/extract_KeyNet/model/HyNet/weights/HyNet_LIB.pth',

                    # Extraction Parameters
                    'nms_size': 15,
                    'pyramid_levels': 4,
                    'up_levels': 1,
                    'scale_factor_levels': np.sqrt(2),
                    's_mult': 22,
                },
        }
        conf = keynet_config['KeyNet_default_config']
        keynet_model, desc_model = initialize_networks(conf)

        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        self.keynet_model = keynet_model
        self.desc_model = desc_model
        self.conf = conf
        self.device =device
        self.num_kpts = num_kpts

    def __call__(self, path):
        xys, desc =  compute_kpts_desc(path, self.keynet_model, self.desc_model, self.conf, self.device, self.num_kpts)

        return xys, desc


if __name__ == "__main__":
    det = KeyNet_extractor(num_kpts=1500)

    kp, desc= det("../temp.jpg")
    print(kp.shape,desc.shape)

