import open3d as o3d
import os
import numpy as np
import h5py
# import MinkowskiEngine as ME

####################################### io (only geometry) #######################################
def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:,0:3].astype('int32')

    return coords

def write_h5_geo(filedir, coords):
    data = coords.astype('int32')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int32')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int32')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def read_ply_o3d_geo(filedir, dtype='int32'):
    pcd = o3d.io.read_point_cloud(filedir)
    coords = np.asarray(pcd.points).astype(dtype)

    return coords

def write_ply_o3d_geo(filedir, coords, dtype='int32'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    fo = open(filedir, "w")
    fo.writelines(lines)

    return

def write_ply_o3d_normal(filedir, coords, dtype='int32', knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(dtype))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    o3d.io.write_point_cloud(filedir, pcd, write_ascii=True)
    f = open(filedir)
    lines = f.readlines()
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'
    lines[7] = 'property float nx\n'
    lines[8] = 'property float ny\n'
    lines[9] = 'property float nz\n'
    fo = open(filedir, "w")
    fo.writelines(lines)

    return

####################################### io #######################################
def read_h5(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('uint8')

    return coords, feats

def write_h5(filedir, coords, feats):
    coords = coords.astype('int16')
    feats = feats.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('coords', data=coords, shape=coords.shape)
        h.create_dataset('feats', data=feats, shape=feats.shape)
    # print(feats.mean().round(), feats.max(), feats.min())

    return

def read_h5_label(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('uint8')
    label = h5py.File(filedir, 'r')['label'][:].astype('uint8')
    if False: print('DBG read h5 mae:', np.abs(feats.astype('float') - label.astype('float')).mean().round(2))
    

    return coords, feats, label

def read_h5_gt(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('uint8')
    gt_coords = h5py.File(filedir, 'r')['gt_coords'][:].astype('int16')
    gt_feates = h5py.File(filedir, 'r')['gt_feates'][:].astype('uint8')
    if False: print('DBG read h5 mae:', np.abs(feats.astype('float') - gt_feates.astype('float')).mean().round(2))

    return coords, feats, gt_coords, gt_feates

def write_h5_label(filedir, coords, feats, label):
    coords = coords.astype('int16')
    feats = feats.astype('uint8')
    label = label.astype('uint8')
    if False: print('DBG write h5 mae:', np.abs(feats.astype('float') - label.astype('float')).mean().round(2))
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('coords', data=coords, shape=coords.shape)
        h.create_dataset('feats', data=feats, shape=feats.shape)
        h.create_dataset('label', data=label, shape=label.shape)
    # print(feats.mean().round(), feats.max(), feats.min())

    return

def read_ply_old_ascii(filedir, order='rgb'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('float32')
    if data.shape[-1]==6: feats = data[:,3:6].astype('uint8')
    if data.shape[-1]>6: 
        feats = data[:, 6:9]
        if np.all(feats == np.floor(feats)):
            feats = data[:, 6:9].astype('uint8')
        else:
            feats = data[:, 3:6].astype('uint8')
    if order=='gbr': feats = np.hstack([feats[:,2:3], feats[:,0:2]])
    return coords, feats


def read_ply_ascii(filedir, order='rgb'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('float32')
    if data.shape[-1]==6: feats = data[:,3:6].astype('float32')
    if data.shape[-1]>6:
        feats = data[:, 6:9]
        if np.all(feats == np.floor(feats)):
            feats = data[:, 6:9].astype('uint8')
        else:
            feats = data[:, 3:6].astype('uint8')
    if order=='gbr': feats = np.hstack([feats[:,2:3], feats[:,0:2]])
    coords = torch.tensor(coords).float()
    feats = (torch.tensor(feats).float() / 255.)
    return coords, feats

####################################### partition #######################################
def kdtree_partition(pc, max_num):
    parts = []
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]  
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        return root
    init_root = KD_node(None)
    root = createKDTree(init_root, pc)  

    return parts

####################################### reorder #######################################
def sort_points(coords, feats):
    indices_sort = np.argsort(array2vector(coords))

    return coords[indices_sort], feats[indices_sort]


def array2vector(array):
    # 3D -> 1D by sum each dimension
    array = array.astype('int64')
    step = array.max() + 1
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector

def array2vector_torch(array, step):
    array = array.long().cpu()
    step = array.max()+1
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector


####################################### color space conversion #######################################
############## BT.601 ##############
# Y = 0.257*R + 0.504*G + 0.098*B + 16
# Cb = -0.148*R - 0.291*G + 0.439*B + 128
# Cr = 0.439*R - 0.368*G - 0.071*B + 128
## (235, 16), (240, 16), (240, 16)
# R = 1.164*(Y-16) + 1.596*(Cr-128)
# G = 1.164*(Y-16) - 0.813*(Cr-128) - 0.392*(Cb-128)
# B = 1.164*(Y-16) + 2.017*(Cb-128)

def ycgcor2rgb(ycgco):

    rgb = ycgco.clone()
    rgb[:, 0] = 1 * ycgco[:, 0] + 0.5 * ycgco[:, 2] - 0.5 * ycgco[:, 1]
    rgb[:, 1] = 1 * ycgco[:, 0] + 0.5  * ycgco[:, 1]
    rgb[:, 2] = 1 * ycgco[:, 0] - 0.5 * ycgco[:, 1] - 0.5 * ycgco[:, 2]


    return rgb

def rgb2ycgcor(rgb):
    ycgco = rgb.clone()
    ycgco[:, 0] = 0.25*rgb[:,0] + 0.5*rgb[:,1] + 0.25*rgb[:,2]
    ycgco[:, 1] = -0.5 * rgb[:, 0] + 1 * rgb[:, 1] - 0.5 * rgb[:, 2]
    ycgco[:, 2] = 1 * rgb[:, 0] + 0 * rgb[:, 1] - 1 * rgb[:, 2]

    return ycgco

def ycgco2rgb(ycgco):

    rgb = ycgco.clone()
    rgb[:, 0] = 1 * ycgco[:, 0] - 1 * ycgco[:, 1] + 1 * ycgco[:, 2]
    rgb[:, 1] = 1 * ycgco[:, 0] + 1 * ycgco[:, 1]
    rgb[:, 2] = 1 * ycgco[:, 0] - 1 * ycgco[:, 1] - 1 * ycgco[:, 2]

    return rgb

def ycgco2yuv(ycgco):
    rgb = ycgco2rgb(ycgco)
    yuv = rgb2yuv(rgb)
    return yuv

def rgb2ycgco(rgb):
    ycgco = rgb.clone()
    ycgco[:, 0] = 0.25*rgb[:,0] + 0.5*rgb[:,1] + 0.25*rgb[:,2]
    ycgco[:, 1] = -0.25 * rgb[:, 0] + 0.5 * rgb[:, 1] - 0.25 * rgb[:, 2]
    ycgco[:, 2] = 0.5 * rgb[:, 0] + 0 * rgb[:, 1] - 0.5 * rgb[:, 2]

    return ycgco

def rgb2yuv(rgb):
    rgb = 255 * rgb
    yuv = rgb.clone()
    yuv[:,0] = 0.257*rgb[:,0] + 0.504*rgb[:,1] + 0.098*rgb[:,2] + 16
    yuv[:,1] = -0.148*rgb[:,0] - 0.291*rgb[:,1] + 0.439*rgb[:,2] + 128
    yuv[:,2] = 0.439*rgb[:,0] - 0.368*rgb[:,1] - 0.071*rgb[:,2] + 128
    yuv[:,0] = (yuv[:,0]-16)/(235-16)
    yuv[:,1] = (yuv[:,1]-16)/(240-16)
    yuv[:,2] = (yuv[:,2]-16)/(240-16)
    return yuv
#

# def rgb2yuv(rgb):
#     """input: [0,1];    output: [0,1]
#     """
#     rgb = 255*rgb
#     yuv = rgb.clone()
#     yuv[:,0] = 0.257*rgb[:,0] + 0.504*rgb[:,1] + 0.098*rgb[:,2] + 16
#     yuv[:,1] = -0.148*rgb[:,0] - 0.291*rgb[:,1] + 0.439*rgb[:,2] + 128
#     yuv[:,2] = 0.439*rgb[:,0] - 0.368*rgb[:,1] - 0.071*rgb[:,2] + 128
#     yuv[:,0] = (yuv[:,0]-16)/(235-16)
#     yuv[:,1] = (yuv[:,1]-16)/(240-16)
#     yuv[:,2] = (yuv[:,2]-16)/(240-16)
#
#     return yuv
#
def yuv2rgb(yuv):
    """input: [0,1];    output: [0,1]
    """
    yuv[:,0] = (235-16)*yuv[:,0]+16
    yuv[:,1] = (240-16)*yuv[:,1]+16
    yuv[:,2] = (240-16)*yuv[:,2]+16
    rgb = yuv.clone()
    rgb[:,0] = 1.164*(yuv[:,0]-16) + 1.596*(yuv[:,2]-128)
    rgb[:,1] = 1.164*(yuv[:,0]-16) - 0.813*(yuv[:,2]-128) - 0.392*(yuv[:,1]-128)
    rgb[:,2] = 1.164*(yuv[:,0]-16) + 2.017*(yuv[:,1]-128)
    rgb = rgb/255
    
    return rgb

def mse_yuv(dataA, dataB, weight):
    MSELoss = torch.nn.MSELoss().to(dataA.device)
    dataA = rgb2yuv(dataA)
    dataB = rgb2yuv(dataB)
    mse = (weight*MSELoss(dataA[:,0], dataB[:,0]) + \
            MSELoss(dataA[:,1], dataB[:,1]) + \
            MSELoss(dataA[:,2], dataB[:,2]))/(weight+2)

    return mse


def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    print(data.shape)
    print(groud_truth.shape)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(data.F)
    mask = get_target_by_sp_tensor(data, groud_truth.C.cpu()).cuda()
    print(mask[mask==True].shape)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    # bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    # sum_bce = bce * data.shape[0]

    return bce


####################################### sparse tensor processing #######################################
import torch
import MinkowskiEngine as ME

def get_target_by_sp_tensor(out, coords_T):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 0] \
                         + coords[:, 1] * step \
                         + coords[:, 2] * step * step \
                         + coords[:, 3] * step * step * step
            return coords_sum

        step = max(out.C.cpu().max(), coords_T.max()) + 1
        out_sp_tensor_coords_1d = ravel_multi_index(out.C.cpu(), step)
        target_coords_1d = ravel_multi_index(coords_T, step)
        # test whether each element of a 1-D array is also present in a second array.
        target = np.in1d(out_sp_tensor_coords_1d, target_coords_1d)

        return torch.Tensor(target).bool()
def isin(data, ground_truth):
    """ Input data and ground_truth are torch tensor of shape [N, D].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is in `ground_truth` and False otherwise.
    """
    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1 #求通道数
    data = array2vector_torch(data, step)
    ground_truth = array2vector_torch(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def istopk(data, nums, rho=1.0):
    """ Input data is sparse tensor and nums is a list of shape [batch_size].
    Returns a boolean vector of the same length as `data` that is True
    where an element of `data` is the top k (=nums*rho) value and False otherwise.
    """
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)

def load_sparse_tensor(filedir, max_num_points=1e8, device='cuda', order='rgb'):
    if filedir.endswith('h5'): coords, feats = read_h5(filedir)
    if filedir.endswith('ply'): coords, feats = read_ply_old_ascii(filedir, order=order)
    if coords.shape[0] <= max_num_points:      
        coords = torch.tensor(coords).int()
        feats = torch.tensor(feats).float() / 255.
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
        return x

    else:
        points = np.hstack([coords.astype('int16'), feats.astype('int16')])
        points_list = kdtree_partition(points, max_num=max_num_points)
        x_list = []
        for points_part in points_list:
            # print('DBG!!! ', points_part.shape)
            coords_part = torch.tensor(points_part[:, 0:3]).int()
            feats_part = torch.tensor(points_part[:, 3:6]).float()/255.
            # print('DBG!!! ', coords_part.shape, feats_part.shape)
            coords_part, feats_part = ME.utils.sparse_collate([coords_part], [feats_part])
            x_part = ME.SparseTensor(features=feats_part, coordinates=coords_part, tensor_stride=1, device=device)
            x_list.append(x_part)
            
        return x_list

def sort_sparse_tensor(sparse_tensor):
    """ Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu().numpy()))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)

    return sparse_tensor_sort

def sort_tensor(coords, feats):
    indices_sort = np.argsort(array2vector(coords.cpu().numpy()))
    sort_feats = feats[indices_sort]
    sort_coords = coords[indices_sort]

    return sort_coords, sort_feats


def load_coords_sparse(file, device):
    coords = read_ply_ascii_geo(file)
    feats = torch.ones((len(coords), 1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    coords_sparse = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    return coords_sparse

def load_h5_sparse_tensor(input_file, gt_file,  device='cuda', order='rgb'):
    coords, feats = read_ply_ascii(input_file, order=order)
    gt_coords, gt_feats = read_ply_ascii(gt_file)
    coords = torch.tensor(coords).int()
    feats = torch.tensor(feats).float() / 255.
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    gt_coords = torch.tensor(gt_coords).int()
    gt_feats = torch.tensor(gt_feats).float() / 255.
    gt_coords, gt_feats = ME.utils.sparse_collate([gt_coords], [gt_feats])
    gt = ME.SparseTensor(features=gt_feats, coordinates=gt_coords, tensor_stride=1, device=device)

    x = sort_sparse_tensor(x)
    gt = sort_sparse_tensor(gt)
    return x, gt

def has_decimal(tensor: torch.Tensor, eps: float = 1e-6) -> bool:
    diff = (tensor - tensor.round()).abs()
    return (diff > eps).any().item()
