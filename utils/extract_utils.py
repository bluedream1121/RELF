import os, cv2, json, torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F 

def load_images(list_images_txt, split_path, split):
    splits = json.load(open(split_path))
    target_scenes= splits[split]['test']

    f = open(list_images_txt, "r")
    image_list = []
    for path_to_image in sorted(f.readlines()):
        scene_name = path_to_image.split('/')[-2]
        if scene_name not in target_scenes:
            continue
        image_list.append(path_to_image)
    return sorted(image_list)

def get_image_list(dataset_path, dataset):

    if dataset == 'hpatches':
        list_images_txt='{}/hpatches-sequences-release/HPatches_images.txt'.format(dataset_path)
        split_path = '{}/hpatches-sequences-release/splits.json'.format(dataset_path)
        split = 'full'
    if dataset == 'hpatches_val':
        list_images_txt='{}/hpatches-sequences-release/HPatches_images.txt'.format(dataset_path)
        split_path = '{}/hpatches-sequences-release/splits.json'.format(dataset_path)
        split = 'debug'        
    elif dataset == 'roto360':
        list_images_txt='{}/roto360/HPatches_rot_image.txt'.format(dataset_path)
        split_path = '{}/roto360/splits.json'.format(dataset_path)
        split = 'debug'

    image_list = load_images(list_images_txt, split_path, split)

    return image_list




def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)

def check_directory(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def interpolate_feats(img,pts,feats):
    """
    img : B, 3, H, W
    pts : B, N, 2
    feats : B, C, H', W'
    """
    # compute location on the feature map (due to pooling)
    _, _, h, w = feats.shape
    pool_num = img.shape[-1] / feats.shape[-1]
    pts_warp= pts / pool_num
    pts_norm=normalize_coordinates(pts_warp,h,w)
    pts_norm=torch.unsqueeze(pts_norm, 1)  # b,1,n,2

    # interpolation
    pfeats=F.grid_sample(feats, pts_norm, 'bilinear', align_corners=True)[:, :, 0, :]  # b,f,n
    pfeats=pfeats.permute(0,2,1) # b,n,f

    return pfeats

def normalize_coordinates(coords, h, w):
    h=h-1
    w=w-1
    coords=coords.clone().detach()
    coords[:, :, 0]-= w / 2
    coords[:, :, 1]-= h / 2
    coords[:, :, 0]/= w / 2
    coords[:, :, 1]/= h / 2
    return coords

    

