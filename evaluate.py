import os, cv2, tqdm, torch
from config import get_config
from utils.logger import Logger, AverageMeterMatching
from model.descriptor_utils import DescGroupPoolandNorm
from utils.extract_utils import get_image_list
from utils.evaluate_utils import *

from baselines.extract_SuperPoint.get_kpts_desc import SuperPoint_extrator
from baselines.extract_GIFT.get_kpts_desc import GIFT_SuperPoint
from baselines.extract_KeyNet.get_kpts_desc import KeyNet_extractor
from baselines.extract_SIFT.get_kpts_desc import SIFT_detector


from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from model.load_model import load_model


class EvaluatePlanarScenes:
    def __init__(self, args):
        dataset_path = 'data'

        ## define the keypoint detector
        if args.detector == 'sift':
            det = SIFT_detector()         
        elif args.detector == 'gift':
            det = GIFT_SuperPoint()
        elif args.detector == 'superpoint':
            det = SuperPoint_extrator()
        elif args.detector == 'keynet':
            det = KeyNet_extractor(num_kpts=1000)

        image_list =  get_image_list(dataset_path, args.eval_dataset)

        ## Image loader
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.eval_dataset = args.eval_dataset
        self.dataset_path = dataset_path
        self.image_list = image_list
        self.transform = transform
        self.detector = args.detector
        self.det = det

        self.pool_and_norm = DescGroupPoolandNorm(args)

    def __call__(self, model):
        with torch.no_grad():
            average_meter_match = AverageMeterMatching()
            for ii, imname in tqdm.tqdm(enumerate(self.image_list), total=len(self.image_list)):

                if self.check_source(imname):
                    src_kpts, src_descs, src_imname = self.extract_descriptors(imname, model)
                    continue

                trg_kpts, trg_descs, trg_imname = self.extract_descriptors(imname, model)

                k1, k2, d1, d2 = self.descriptor_pooling(src_kpts, trg_kpts, src_descs, trg_descs, model == None)

                H, angle = self.get_gt_homography(src_imname, trg_imname)

                matches, distances, total_points = self.compute_matches(k1, k2, d1, d2, H)

                average_meter_match.update(angle, matches.shape[0], distances, total_points)

        return average_meter_match


    def get_image(self, imname):
        r"""Reads PIL image from path"""
        image = Image.open(imname).convert('RGB')
        image = np.array(image)
        image = self.transform(image)
        return image

    def check_source(self, imname):
        if self.eval_dataset == 'roto360':
            return '_rot0.jpg' in imname
        elif self.eval_dataset == 'hpatches' or  self.eval_dataset == 'hpatches_val':
            return '1.ppm' in imname 
        else:
            raise NotImplementedError

    def detect_keypoints(self, src_imname):
        source_image = cv2.imread(src_imname)

        if self.detector == 'superpoint' or self.detector == 'keynet' \
             or self.detector == 'sift' or self.detector == 'gift' :
            kpts, descs = self.det(src_imname)
        else:
            raise NotImplementedError

        return kpts[:, :2], descs

    def extract_descriptors(self, imname, model):
        imname = os.path.join(self.dataset_path, imname.rstrip())
        image = self.get_image(imname)

        ## treat high-resolution
        if max(image.shape) > 3000:
            image = F.interpolate(image.unsqueeze(0), scale_factor=0.5 , mode='bilinear', align_corners=True).squeeze(0)
        
        kpts, descs = self.detect_keypoints(imname)
        if model != None:
            descs = model(image.unsqueeze(0).float().cuda(), torch.from_numpy(kpts).unsqueeze(0).float().cuda())

        return kpts[:, :2], descs, imname

    def descriptor_pooling(self, src_kpts, trg_kpts, src_descs, trg_descs, baseline=False):
        if baseline:
            ### A. baseline evaluation
            k1 = src_kpts[:, :2]; k2 = trg_kpts[:, :2]
            d1 = src_descs 
            d2 = trg_descs 
        else:
            ### B. ours evaluation
            k1, d1 = self.pool_and_norm.desc_pool_and_norm_infer(torch.tensor(src_kpts).unsqueeze(0), src_descs)
            k2, d2 = self.pool_and_norm.desc_pool_and_norm_infer(torch.tensor(trg_kpts).unsqueeze(0), trg_descs)
            k1 = k1[0]; k2 = k2[0]
            d1 = d1[0]; d2 = d2[0]

        return k1, k2, d1, d2

    def get_gt_homography(self, src_imname, trg_imname):
        if self.eval_dataset == 'roto360':
            H = GetGroundTruthSytheData(trg_imname)
            angle = trg_imname.split("rot")[-1][:-4].zfill(3)
        elif self.eval_dataset == 'hpatches' or  self.eval_dataset == 'hpatches_val':
            H = GetGroundTruth(src_imname, trg_imname)
            angle = src_imname.split("/")[-2][0]    ## this value is not angle, but logging key.
        else:
            raise NotImplementedError
        return H, angle

    def compute_matches(self, k1, k2, d1, d2, H):
        total_points = (len(d1) + len(d2)) / 2

        matches, _ = mnn_matcher(d1, d2)

        k1_match = k1[matches[:, 0], :2]#.cpu().numpy()
        k2_match = k2[matches[:, 1], :2]#.cpu().numpy()

        distances = compute_correctness(k1_match, k2_match, H)

        return matches, distances, total_points


if __name__ == "__main__":
    args = get_config()
    logger = Logger.initialize(args, training=False)

    print(args)

    ## Load model
    model = load_model(args)

    evaluator = EvaluatePlanarScenes(args)
    result = evaluator(model)

    result.print_results(logger, printing='console')



