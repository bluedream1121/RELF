import argparse, json, os, tqdm, cv2, time
from .demo_superpoint import SuperPointFrontend
import numpy as np

def check_directory(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)

def read_image(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
        img_size: (W, H) tuple specifying resize size.
    Returns
        grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

def read_image_wo_resize(impath):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
    Returns
        grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    grayim = (grayim.astype('float32') / 255.)
    return grayim


class SuperPoint_extrator:
    def __init__(self, weights_path='superpoint_v1.pth', H=120, W=160, nms_dist=4, conf_thresh=0.015, \
                        nn_thresh=0.7, network_version='SuperPoint'):

        weights_path = 'baselines/extract_SuperPoint/superpoint_v1.pth'

        print('==> Loading pre-trained network.')
        fe = SuperPointFrontend(weights_path=weights_path,
                        nms_dist=nms_dist,
                        conf_thresh=conf_thresh,
                        nn_thresh=nn_thresh,
                        cuda=True)
        print('==> Successfully loaded pre-trained network.')

        self.fe = fe
    
    def __call__(self, path):
        # img = read_image(path, (opt.W, opt.H))
        img = read_image_wo_resize(path)

        H, W = img.shape
        # Get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = self.fe.run(img)
        end1 = time.time()

        im_pts = pts.T[:1500]
        descriptors = desc.T[:1500]

        # print(im_pts.shape, descriptors.shape)
        return im_pts, descriptors


if __name__ == '__main__':

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--H', type=int, default=120,
        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
        help='Input image width (default:160).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--network_version', type=str, default='SuperPoint',
        help='')

    opt = parser.parse_args()
    print(opt)

    # This class helps load input images from different sources.
    #   vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=True)
    print('==> Successfully loaded pre-trained network.')

    split = 'full'
    list_images_txt='../../HSequences_bench/HPatches_images.txt'
    split_path = '../../HSequences_bench/splits.json'
    splits = json.load(open(split_path))
    target_scenes= splits[split]['test']
    results_dir='extracted_features/'

    f = open(list_images_txt, "r")
    image_list = []
    for path_to_image in sorted(f.readlines()):
        scene_name = path_to_image.split('/')[-2]
        if scene_name not in target_scenes:
            continue
        image_list.append(path_to_image)
    
    # print(image_list)
    print('==> Successfully load HPatches image')

    save_feat_dir = os.path.join(results_dir, opt.network_version)

    check_directory(results_dir)
    check_directory(save_feat_dir)

    iterate = tqdm.tqdm(image_list, total=len(image_list), desc="SuperPoint HPatches")
    feat = {}
    for path_to_image in iterate:

        path = path_to_image.rstrip('\n')
        create_result_dir(os.path.join(save_feat_dir, path))

        print(path)
        if not os.path.exists(path):
            print('[ERROR]: File {0} not found!'.format(path))
            exit()
        
        # img = read_image(path, (opt.W, opt.H))
        img = read_image_wo_resize(path)

        H, W = img.shape
        # Get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()

        im_pts = pts.T[:1500]
        descriptors = desc.T[:1500]

        print(im_pts.shape, descriptors.shape)

        iterate.set_description("Extract {} (H, W) : ({}, {}) -> time {:.4f}s".format('/'.join(path.split('/')[-2:]),  H, W, end1-start1 ))


        ## save the kpts and descriptors
        print("save", os.path.join(save_feat_dir, path)+'.kpt')
        file_name = os.path.join(save_feat_dir, path)+'.kpt'
        np.save(file_name, im_pts)
        file_name = os.path.join(save_feat_dir, path)+'.dsc'
        np.save(file_name, descriptors)
