# file: all_file.py

import os
import open3d
import glob
import os.path

def ply_ascii2bin(file_full_name):
    pointcloud_in = open3d.io.read_point_cloud(file_full_name)
    # write_point_cloud(filename, pointcloud, write_ascii=False, compressed=False,
    #                   print_progress=False):  # real signature unknown; restored from __doc__
    open3d.io.write_point_cloud(filename=file_full_name, pointcloud=pointcloud_in, write_ascii=True)
    print("transformed " + file_full_name + " from ascii to binary")




def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            # ext_str = f.split(".")[1]
            # if ext_str != "ply":
            #     print(f + "不是ply文件")
            #     yield fullname
            #     continue
            fullname = os.path.join(root, f)
            yield fullname


if __name__ == '__main__':

    file_rootdir = '/Elements/PCQA/ICIP_dataset/sampling'
    input_filedirs = sorted(glob.glob(os.path.join(file_rootdir, f'*' + 'ply'), recursive=True))

    for idx, filename in enumerate(input_filedirs):
        ply_ascii2bin(file_full_name=filename)