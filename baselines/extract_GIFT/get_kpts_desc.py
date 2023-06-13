import argparse
from .train.evaluation import EvaluationWrapper

def eval_hpatches():
    evaluator=EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
    evaluator.extract()


class GIFT_SuperPoint:
    def __init__(self, det_cfg='configs/eval/superpoint_det.yaml', \
                desc_cfg='configs/eval/gift_pretrain_desc.yaml', \
                match_cfg='configs/eval/match_v2.yaml'):
        
        det_cfg='baselines/extract_GIFT/configs/eval/superpoint_det.yaml'
        # det_cfg='baselines/extract_GIFT/configs/eval/sift_det.yaml'
        desc_cfg='baselines/extract_GIFT/configs/eval/gift_pretrain_desc.yaml'
        # desc_cfg='baselines/extract_GIFT/configs/eval/none_desc.yaml'        
        # desc_cfg='baselines/extract_GIFT/configs/eval/superpoint_desc.yaml'        
        match_cfg='baselines/extract_GIFT/configs/eval/match_v2.yaml'

        self.evaluator=EvaluationWrapper(det_cfg, desc_cfg, match_cfg)

    def __call__(self, path):
        kpts, desc = self.evaluator.extract(path)
        return kpts, desc

# def get_one_sample():
#     evaluator=EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
#     kpts, desc = evaluator.extract('/home/jongmin/Desktop/temp.jpg')

#     print(kpts.shape, desc.shape)

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='eval')
    # parser.add_argument('--cfg', type=str, default='configs/GIFT-stage1.yaml')
    parser.add_argument('--det_cfg',type=str,default='configs/eval/superpoint_det.yaml')
    parser.add_argument('--desc_cfg',type=str,default='configs/eval/gift_pretrain_desc.yaml')
    parser.add_argument('--match_cfg',type=str,default='configs/eval/match_v2.yaml')
    flags=parser.parse_args()

    # eval_hpatches()
    # get_one_sample()
    extractor = GIFT_SuperPoint()
    kpts, desc = extractor('/home/jongmin/Desktop/temp.jpg')
    print(kpts)
    print(kpts.shape, desc.shape)

    import matplotlib.pyplot as plt
    import cv2
    plt.imshow(cv2.imread('/home/jongmin/Desktop/temp.jpg'))
    plt.scatter(kpts[:, 0], kpts[:, 1])
    plt.savefig("tmp.jpg")
