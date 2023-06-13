import torch

from model.load_model import load_model
from .model.descriptor_utils import DescGroupPoolandNorm

class ReFTDescriptor:
    def __init__(self, args):

        self.model = load_model(args)
        self.pool_and_norm = DescGroupPoolandNorm(args)

    def __call__(self, image, kpts):
        desc = self.model(image, kpts)

        ## kpts torch.tensor ([B, K, 2]), desc torch.tensor ([B, K, CG])
        k1, d1 = self.pool_and_norm.desc_pool_and_norm_infer(kpts, desc)

        return k1, d1



if __name__ == "__main__":

    import os, cv2, torch
    from torchvision import transforms
    from config import get_config
    from baselines.extract_GIFT.utils.superpoint_utils import SuperPointWrapper

    args = get_config()
    extractor = ReFTDescriptor(args)

    det = SuperPointWrapper()

    image_np = cv2.imread("/home/jongmin/Desktop/temp.jpg")
    image = transforms.ToTensor()(image_np)
    print(image.shape)

    kpts, desc = det(image_np)

    k1, d1 = extractor(image.unsqueeze(0).float().cuda(), torch.from_numpy(kpts).unsqueeze(0).float().cuda())
    
    print(kpts.shape, desc.shape)
    print(k1.shape, d1.shape)
