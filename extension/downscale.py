import shutil

from extension.gpcc import gpcc_encode, gpcc_decode
import os
from data_processing.data_utils import read_ply_ascii, sort_tensor

savepath = 'temp'
os.makedirs(savepath, exist_ok=True)
bin_folder = os.path.join(savepath, 'bin');
os.makedirs(bin_folder, exist_ok=True)
rec_folder = os.path.join(savepath, 'rec');
os.makedirs(rec_folder, exist_ok=True)



def down(gt_file):
    file_name = gt_file.split('/')[-1].split('.')[0]
    posQuantscale_list = [0.95, 0.85, 0.5, 0.125]
    downscale_list = [] 
    for idx_scale in range(len(posQuantscale_list)):
        bin_dir = os.path.join(savepath + '/bin', file_name.split('.')[0] + '.bin')
        rec_dir = os.path.join(savepath + '/ply', file_name.split('.')[0] + '_' +'scale' + str(idx_scale + 1) +'.ply')
        if os.path.isfile(rec_dir):
            downscale_list.append(rec_dir)
        else:
            results_enc = gpcc_encode(gt_file, bin_dir, posQuantscale=posQuantscale_list[idx_scale],
                                transformType=0, qp=4, show=False)
            results_dec = gpcc_decode(bin_dir, rec_dir, show=False)
            downscale_list.append(rec_dir)
    scale1_coords, scale1_feats = read_ply_ascii(downscale_list[0])
    scale2_coords, scale2_feats = read_ply_ascii(downscale_list[1])
    scale3_coords, scale3_feats = read_ply_ascii(downscale_list[2])
    scale4_coords, scale4_feats = read_ply_ascii(downscale_list[3])
    scale1_coords, scale1_feats = sort_tensor(scale1_coords, scale1_feats)
    scale2_coords, scale2_feats = sort_tensor(scale2_coords, scale2_feats)
    scale3_coords, scale3_feats = sort_tensor(scale3_coords, scale3_feats)
    scale4_coords, scale4_feats = sort_tensor(scale4_coords, scale4_feats)
    return (scale1_coords, scale1_feats), (scale2_coords, scale2_feats), (scale3_coords, scale3_feats), (scale4_coords, scale4_feats)
