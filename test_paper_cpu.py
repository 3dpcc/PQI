import argparse
import random
import subprocess
import time

import numpy as np
import torch
from data_processing.data_utils import sort_sparse_tensor, load_sparse_tensor, sort_tensor, has_decimal
from data_processing.data_utils import read_ply_ascii
import MinkowskiEngine as ME
import os, glob
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree
from extension.downscale import down
def multiscale(ground_truth, gt_file, is_voxel):
    if is_voxel:
        avg_ds = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        scale1 = avg_ds(ground_truth)
        scale2 = avg_ds(scale1)
        scale3 = avg_ds(scale2)
        scale4 = avg_ds(scale3)
        scale1 = sort_sparse_tensor(scale1)
        scale2 = sort_sparse_tensor(scale2)
        scale3 = sort_sparse_tensor(scale3)
        scale4 = sort_sparse_tensor(scale4)
        return scale1, scale2, scale3, scale4
    else:
        scale1, scale2, scale3, scale4 = down(gt_file)
        return scale1, scale2, scale3, scale4
    


def knn_search(scale,
                       ground_truth_coords, ground_truth_feats,
                       distortion_pc_coords, distortion_pc_feats, k):
    # 张量 → numpy
    coords_k = scale.C[:,1:].cpu().numpy()          # (N_k,3)
    coords_d = distortion_pc_coords.cpu().numpy()    # (N_d,3)
    feats_d = distortion_pc_feats.cpu().numpy()      # (N_d,F)
    # 构建树并查询
    tree = cKDTree(coords_d)
    dists_dd, idx_dd = tree.query(coords_k, k=k)
    # B→1 维度补回 Torch
    idx_dd = torch.from_numpy(idx_dd).long().to(distortion_pc_coords.device)       # (N_k, K)
    dists_dd = torch.from_numpy(dists_dd).float().to(distortion_pc_coords.device)   # (N_k, K)
    coords_dd = distortion_pc_coords[idx_dd]                                       # (N_k, K, 3)
    feats_dd  = distortion_pc_feats[idx_dd]                                        # (N_k, K, F)

    # 同理对 ground_truth 做一次
    coords_gt = ground_truth_coords.cpu().numpy()
    feats_gt  = ground_truth_feats.cpu().numpy()
    tree2 = cKDTree(coords_gt)
    dists_dr, idx_dr = tree2.query(coords_k, k=k)
    idx_dr = torch.from_numpy(idx_dr).long().to(distortion_pc_coords.device)
    dists_dr = torch.from_numpy(dists_dr).float().to(distortion_pc_coords.device)
    coords_dr = ground_truth_coords[idx_dr]
    feats_dr  = ground_truth_feats[idx_dr]

    # 恢复 batch 维度 (B=1)
    return {
      'coords_dd': coords_dd.unsqueeze(0),
      'coords_dr': coords_dr.unsqueeze(0),
      'feats_dd' : feats_dd.unsqueeze(0),
      'feats_dr' : feats_dr.unsqueeze(0),
      'dists_dd' : dists_dd.unsqueeze(0),
      'dists_dr' : dists_dr.unsqueeze(0),
      'coords_d' : distortion_pc_coords.unsqueeze(0),
      'feats_d'  : distortion_pc_feats.unsqueeze(0),
    }

def knn_search_not_voxel(scale,
                       ground_truth_coords, ground_truth_feats,
                       distortion_pc_coords, distortion_pc_feats, k):
    coords_k = scale.cpu().numpy()          # (N_k,3)
    coords_d = distortion_pc_coords.cpu().numpy()    # (N_d,3)
    feats_d = distortion_pc_feats.cpu().numpy()      # (N_d,F)
    tree = cKDTree(coords_d)
    dists_dd, idx_dd = tree.query(coords_k, k=k)
    idx_dd = torch.from_numpy(idx_dd).long().to(distortion_pc_coords.device)       # (N_k, K)
    dists_dd = torch.from_numpy(dists_dd).float().to(distortion_pc_coords.device)   # (N_k, K)
    coords_dd = distortion_pc_coords[idx_dd]                                       # (N_k, K, 3)
    feats_dd  = distortion_pc_feats[idx_dd]                                        # (N_k, K, F)

    coords_gt = ground_truth_coords.cpu().numpy()
    feats_gt  = ground_truth_feats.cpu().numpy()
    tree2 = cKDTree(coords_gt)
    dists_dr, idx_dr = tree2.query(coords_k, k=k)
    idx_dr = torch.from_numpy(idx_dr).long().to(distortion_pc_coords.device)
    dists_dr = torch.from_numpy(dists_dr).float().to(distortion_pc_coords.device)
    coords_dr = ground_truth_coords[idx_dr]
    feats_dr  = ground_truth_feats[idx_dr]

    # 恢复 batch 维度 (B=1)
    return {
      'coords_dd': coords_dd.unsqueeze(0),
      'coords_dr': coords_dr.unsqueeze(0),
      'feats_dd' : feats_dd.unsqueeze(0),
      'feats_dr' : feats_dr.unsqueeze(0),
      'dists_dd' : dists_dd.unsqueeze(0),
      'dists_dr' : dists_dr.unsqueeze(0),
      'coords_d' : distortion_pc_coords.unsqueeze(0),
      'feats_d'  : distortion_pc_feats.unsqueeze(0),
    }



def score(knn_outset, keypoints_feats, k, distance_type):
    constant = 0.001
    keypoints_feats = keypoints_feats.unsqueeze(0).unsqueeze(2)
    dists_dd = knn_outset['dists_dd']  # 1, n, k.unsqueeze(2)
    dists_dr = knn_outset['dists_dr']  # 1, n, k

    if distance_type == '1-norm':
        ref_distance_dif = torch.sum(dists_dr ** 2, 2)
        dis_distance_dif = torch.sum(dists_dd ** 2, 2)
    elif distance_type == '2-norm':
        ref_distance_dif = dists_dr ** 2
        dis_distance_dif = dists_dd ** 2

    feats_dd = knn_outset['feats_dd']
    feats_dr = knn_outset['feats_dr']

    ref_color_dif = feats_dr - keypoints_feats
    dis_color_dif = feats_dd - keypoints_feats
    ref_distance_dif_mean = torch.mean(ref_distance_dif, dim=2)
    dis_distance_dif_mean = torch.mean(dis_distance_dif, dim=2)

    ref_color_dif_mean = torch.mean(ref_color_dif, dim=2)
    dis_color_dif_mean = torch.mean(dis_color_dif, dim=2)

    geo_ssim = (2 * ref_distance_dif_mean * dis_distance_dif_mean + constant) / (
                ref_distance_dif_mean ** 2 + dis_distance_dif_mean ** 2 + constant)
    attr_ssim = (2 * ref_color_dif_mean * dis_color_dif_mean + constant) / (
                ref_color_dif_mean ** 2 + dis_color_dif_mean ** 2 + constant)
    attr_final_ssim = (1 * attr_ssim[:,:,0] + 2 * attr_ssim[:,:,1] + 1 * attr_ssim[:,:,2]) / 4
    ssim_score = geo_ssim + attr_final_ssim
    geo_final_score = torch.mean(geo_ssim).item()
    attr_final_score = torch.mean(attr_final_ssim).item()
    final_score = torch.mean(ssim_score).item()

    return final_score, geo_final_score, attr_final_score




def score_point(ground_truth, distortion_pc, k, scale1, scale2, scale3, scale4):

    distortion_pc_coords, distortion_pc_feats = read_ply_ascii(distortion_pc)
    ground_truth_coords, ground_truth_feats = read_ply_ascii(ground_truth)
    ground_truth_coords, ground_truth_feats = sort_tensor(ground_truth_coords, ground_truth_feats)
    distortion_pc_coords, distortion_pc_feats = sort_tensor(distortion_pc_coords, distortion_pc_feats)
    scale1_knn_outset = knn_search(scale1, ground_truth_coords, ground_truth_feats, distortion_pc_coords, distortion_pc_feats, k)
    scale2_knn_outset = knn_search(scale2, ground_truth_coords, ground_truth_feats, distortion_pc_coords, distortion_pc_feats, k)
    scale3_knn_outset = knn_search(scale3, ground_truth_coords, ground_truth_feats, distortion_pc_coords, distortion_pc_feats, k)
    scale4_knn_outset = knn_search(scale4, ground_truth_coords, ground_truth_feats, distortion_pc_coords, distortion_pc_feats, k)
    scale1_score, _, _ = score(scale1_knn_outset, scale1.F, k, '2-norm')
    scale2_score, _, _ = score(scale2_knn_outset, scale2.F, k, '2-norm')
    scale3_score, _, _ = score(scale3_knn_outset, scale3.F, k, '2-norm')
    scale4_score, _, _ = score(scale4_knn_outset, scale4.F, k, '2-norm')
    final_score = (scale1_score + scale2_score + scale3_score + scale4_score) / 4

    return final_score


from concurrent.futures import ThreadPoolExecutor

def score_point_parallel(ground_truth, distortion_pc, k, scales):
    # 1. 读入并排序一次
    distortion_pc_coords, distortion_pc_feats = read_ply_ascii(distortion_pc)
    ground_truth_coords,   ground_truth_feats = read_ply_ascii(ground_truth)
    # ground_truth_coords,   ground_truth_feats = sort_tensor(ground_truth_coords,   ground_truth_feats)
    # distortion_pc_coords,  distortion_pc_feats = sort_tensor(distortion_pc_coords,  distortion_pc_feats)

    # 2. 准备参数列表
    knn_args = [
        (scale, ground_truth_coords, ground_truth_feats,
         distortion_pc_coords, distortion_pc_feats, k)
        for scale in scales
    ]
    # 3. 并行 knn_search（结果顺序与 scales 一致）
    with ThreadPoolExecutor(max_workers=len(scales)) as executor:
        knn_results = list(executor.map(lambda args: knn_search(*args), knn_args))

    # 4. 并行 score 计算（同样保持顺序）
    score_args = [
        (knn_results[i], scales[i].F, k, '2-norm')
        for i in range(len(scales))
    ]
    with ThreadPoolExecutor(max_workers=len(scales)) as executor:
        # 每个 result 是 (score_i, _, _)
        score_results = list(executor.map(lambda args: score(*args), score_args))

    # 5. 提取分数并平均
    scores = [res[0] for res in score_results]
    final_score = sum(scores) / len(scores)
    return final_score

def score_point_parallel_not_voxel(ground_truth, distortion_pc, k, scales):
    distortion_pc_coords, distortion_pc_feats = read_ply_ascii(distortion_pc)
    ground_truth_coords,   ground_truth_feats = read_ply_ascii(ground_truth)
    knn_args = [
        (scale[0], ground_truth_coords, ground_truth_feats,
         distortion_pc_coords, distortion_pc_feats, k)
        for scale in scales
    ]
    with ThreadPoolExecutor(max_workers=len(scales)) as executor:
        knn_results = list(executor.map(lambda args: knn_search_not_voxel(*args), knn_args))

    score_args = [
        (knn_results[i], scales[i][1], k, '2-norm')
        for i in range(len(scales))
    ]
    with ThreadPoolExecutor(max_workers=len(scales)) as executor:
        score_results = list(executor.map(lambda args: score(*args), score_args))

    scores = [res[0] for res in score_results]
    final_score = sum(scores) / len(scores)
    return final_score


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt_rootdir", type=str, default='/media/ivc-18/Elements/PCQA/SJTU_dataset/reference')
    parser.add_argument("--test_rootdir", type=str, default='/media/ivc-18/Elements/PCQA/SJTU_dataset/distortion')
    parser.add_argument('--output_rootdir', type=str, default="output/SJTU-PCQA")
    parser.add_argument('--knn', type=int, default=10)
    args = parser.parse_args()

    return args


def set_seed(seed=42):
    # Python 内置随机数
    random.seed(seed)
    # Numpy 随机数
    np.random.seed(seed)
    # PyTorch 随机数
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 环境随机性
    os.environ['PYTHONHASHSEED'] = str(seed)





if __name__ == '__main__':
    args = parse_args()
    set_seed(0)
    gt_filedirs = sorted(glob.glob(os.path.join(args.gt_rootdir, f'*' + 'ply'), recursive=True))
    gt_filedirs = [gt_filedirs[1]]
    os.makedirs(args.output_rootdir,exist_ok=True)
    all_results = {}
    all_results = pd.DataFrame([all_results])
    for idx, gt in enumerate(tqdm(gt_filedirs)):
        filename = gt.split('/')[-1].split('.')[0]
        test_filedirs = sorted(glob.glob(os.path.join(args.test_rootdir, filename, f'*' + 'ply'), recursive=False))
        # print(test_filedirs)
        ground_truth_sparsetensor = load_sparse_tensor(gt, device='cpu')
        ground_truth_sparsetensor = sort_sparse_tensor(ground_truth_sparsetensor)
        ground_truth_coords,   ground_truth_feats = read_ply_ascii(gt)
        is_voxel = has_decimal(ground_truth_coords)
        # is_voxel = (filename != 'Romanoillamp')
        scale1, scale2, scale3, scale4 = multiscale(ground_truth_sparsetensor, gt, is_voxel)
        for i, test in enumerate(tqdm(test_filedirs)):
            print(test.split('/')[-1])
            results = {}
            start = time.time()
            if is_voxel:
                final_score = score_point_parallel(gt, test, args.knn, [scale1, scale2, scale3, scale4])
            else:
                final_score = score_point_parallel_not_voxel(gt, test, args.knn, [scale1, scale2, scale3, scale4])
            process_time = time.time() - start
            print('final_score:\t', final_score)

            results['filename'] = os.path.basename(test).split('.')[0]
            results['Final Score'] = final_score
            results['process_time'] = process_time
            results = pd.DataFrame([results])
            all_results = pd.concat([all_results, results], ignore_index=True)
            all_csv = os.path.join(args.output_rootdir, 'all.csv')
            all_results.to_csv(all_csv, index=False)
