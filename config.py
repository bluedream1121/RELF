import os, argparse
import random, torch, numpy

def get_config():
    ## hyperparameters
    config = argparse.ArgumentParser()
    config.add_argument('--model', type=str, help='The name of the model to use')
    config.add_argument('--num_group', required=False, default=16, type=int)
    config.add_argument('--load_dir', required=False, default='', type=str, help='trained weights load.' )
    config.add_argument('--detector', required=False, default='superpoint', type=str, help='sift, grid, gridGT ...')
    ## for ReResNet
    config.add_argument('--channels', required=False, default=64, type=int)
    ## for training 
    config.add_argument('--num_epochs', required=False, default=20, type=int)
    config.add_argument('--batch_size', required=False, default=8, type=int)
    config.add_argument('--lr', required=False, default=1e-4, type=float)
    config.add_argument('--alpha', required=False, default=10, type=float, help='Loss balance term multiplying to ori_loss.')
    config.add_argument('--training_breaker', required=False, default=1000, type=int, help='training dataset iterator.')
    config.add_argument('--num_keypoints', required=False, default=512, type=int, help='training dataset number of keypoints.')
    ## for evaluation
    config.add_argument('--eval_dataset', required=False, default='roto360', type=str, help='evaluation dataset, roto360, hpatches' )
    config.add_argument('--candidate', required=False, default="top1", type=str, help='topk or [0,1] ratio value.')

    config.add_argument('--multi_gpu', required=False, default='-1', type=str, help='multi-gpu triaining.' )    

    args = config.parse_args()

    random_seed = 1121
    fix_randseed(random_seed)

    args.num_group = int(os.environ["Orientation"]) if "Orientation" in os.environ else args.num_group
    args.logpath = ''  

    return args


def fix_randseed(randseed):
    r"""Fix random seed"""
    random.seed(randseed)
    numpy.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    # torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
