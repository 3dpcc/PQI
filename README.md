# Revisit Point Cloud Quality Assessment: Current Advances and a Multiscale-Inspired Approach

## ✨ Introduction

The demand for full-reference point cloud quality assessment (PCQA) has extended across various point cloud services. Unlike image quality assessment where the reference and distorted images are naturally aligned in coordinate and thus allow the point-to-point (P2P) color assessment, the coordinates and attribute of a 3D point cloud may both suffer from distortion, making the P2P evaluation unsuitable. To address this, PCQA methods usually define a set of key points and construct a neighborhood around each key point for neighbor-to-neighbor (N2N) computation on geometry and attribute. However, state-of-the-art PCQA methods often exhibit limitations in certain scenarios due to insufficient consideration of key points and neighborhoods. To overcome these challenges, this paper proposes PQI, a simple yet efficient metric to index point cloud quality. PQI suggests using scale-wise key points to uniformly capture distortions within a point cloud, along with a mild neighborhood size associated with each key point for compromised N2N computation. To achieve this, PQI employs a multiscale framework to obtain key points, ensuring comprehensive feature representation and distortion detection throughout the entire point cloud. Such a multiscale method merges every eight points into one in the downscaling processing, implicitly embedding neighborhood information into a single point and thereby eliminating the need for an explicitly large neighborhood. Further, within each neighborhood, simple features like geometry Euclidean distance difference and attribute value difference are extracted. Feature similarity is then calculated between the reference and distorted samples at each scale and linearly weighted to generate the final PQI score. Extensive experiments demonstrate the superiority of PQI, consistently achieving state-of-the-art performance across several widely recognized PCQA datasets. Moreover, PQI is highly appealing for practical applications due to its low complexity and flexible scale options


## News
* 2025.06.17 PQI was accepted by TVCG.

## ⚙️ Environment Install 

### Create a conda environment

In addition to the dependency packages originally installed by PQI, **pytorch3d** also needs to be installed.

### Download pytorch

```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```
Also you can use only cpu version
```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```
### Download MinkowskiEngine

```shell
conda install openblas-devel -c anaconda
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
maybe need **pybind11**
```shell
conda install -c conda-forge pybind11
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### Download some components
open3d: ```pip install open3d```

tqdm: ```pip install tqdm```

pandas: ```pip install pandas```

h5py: ```pip install h5py```

scipy: ```pip install scipy```

## 😀 Usage

### Testing
Please download the corresponding dataset and organize the folders in the following structure, using the SJTU-PCQA dataset as an example:

```shell
SJTU-PCQA
├── reference
│   ├── longdress.ply
│   ├── loot.ply
│   ├── ...
├── distortion
│   ├── longdress
│       ├── longdress_00.ply
│       ├── longdress_01.ply
│       ├── ...
│   ├── loot
│       ├── loot_00.ply
│       ├── loot_01.ply
│       ├── ...
│   ├── ...
```
Then, execute the following code:
```shell
python test_paper_point.py --gt_rootdir="./SJTU-PCQA/reference" \
                     --test_rootdir="./SJTU-PCQA/distortion" \
                     --output_rootdir="output/SJTU-PCQA"
```

If some point clouds are stored in BIN format, you can use the following code to convert them to ASCII.
```shell
python bin2ascii.py
```

## Authors
These files are provided by Hangzhou Normal University [3DPCC](https://github.com/3dpcc) and Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). Please contact us (zhangjunzhe@stu.hznu.edu.cn) if you have any questions.

