import os, cv2, json, tqdm, glob, math, torch, argparse
import numpy as np
import pandas as pd


def GetGroundTruth(im1_path, im2_path):
    path = im1_path[:im1_path.rindex(os.sep)]
    im1_name = im1_path[im1_path.rindex(os.sep)+1:].split(".")[0]
    im2_name = im2_path[im2_path.rindex(os.sep)+1:].split(".")[0]
    H_path = path + os.sep + 'H_'+ im1_name + '_' + im2_name

    # load homography matrix
    H = np.fromfile(H_path, sep=" ")
    H.resize((3, 3))

    return H

def GetGroundTruthSytheData(im2_path):
    path = im2_path[:im2_path.rindex(os.sep)]

    im2_name = im2_path[im2_path.rindex(os.sep)+1:].split(".")[0]
    # H_path = path + os.sep + 'H_' + im2_name + '.txt'
    H_path = path + os.sep  + im2_name + '.txt'

    # load homography matrix
    H = np.fromfile(H_path, sep=" ")
    H.resize((3, 3))

    return H

def warpPerspectivePoints(src_points, H):
    # normalize H
    H /= H[2][2]

    ones = np.ones((src_points.shape[0], 1))
    points = np.append(src_points, ones, axis = 1)

    warpPoints = np.dot(H, points.T)
    warpPoints = warpPoints.T / warpPoints.T[:, 2][:,None]

    return warpPoints[:,0:2]


def compute_correctness(k1, k2, H):
    warp_points = warpPerspectivePoints(k1, H) ## warp kpts1 to image2 (using GT)
    gt_k2 = warp_points

    distances = []

    for (x1, y1), (x2, y2) in zip(k2, gt_k2):
        distance = math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)    
        distances.append(distance)

    return distances


################# matchers below ############


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    descriptors_a= descriptors_a.to(device=device)
    descriptors_b= descriptors_b.to(device=device)
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy(), sim



############## compute matching results ################33


def compute_matching_results(results, corr_thres):
    match_cnts = []
    correct_cnts = []
    precisions = []

    for num_matches, distances in zip(results['num_matches'], results['distances']):
        match_cnt = num_matches

        correct_cnt = np.zeros(corr_thres)
        precision = np.zeros(corr_thres)

        for distance in distances:
            for pixel_thres in range(corr_thres):
                if distance <= pixel_thres + 1 :
                    correct_cnt[pixel_thres] += 1

        for pixel_thres in range(corr_thres):
            if match_cnt == 0:
                precision[pixel_thres] = 0
            else:
                precision[pixel_thres] = correct_cnt[pixel_thres] / match_cnt * 100

        match_cnts.append(match_cnt)
        correct_cnts.append(correct_cnt)
        precisions.append(precision)

    match_cnts = np.mean(match_cnts)
    correct_cnts = np.mean(np.array(correct_cnts), axis=0)
    precisions = np.mean(np.array(precisions), axis=0)    
    return match_cnts, correct_cnts, precisions

def print_matching_results(correct_cnts, match_cnts, total_points, precisions):
    index = []
    columns = ['correct matches', 'pred matches', 'total_points', 'MMA']
    data = []

    thresholds = [0,2,4,9]
    for threshold in thresholds:
        index.append("{:d}px".format(threshold+1))
        data.append([correct_cnts[threshold], match_cnts, total_points, precisions[threshold] ])
        # print('{3}, {0:.2f}, {1:.2f}, {2:.2f}'.format( correct_cnts[threshold], match_cnts, precisions[threshold], threshold+1))

    pd.set_option("display.precision", 2)
    df = pd.DataFrame(data=np.array(data), index=index, columns=columns)

    print(df)

