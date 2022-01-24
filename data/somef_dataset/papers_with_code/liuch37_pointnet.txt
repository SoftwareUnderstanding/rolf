# PointNet

PointNet experimentation using PyTorch 1.6.0. Most of the code is taken from [1]. Purpose of this project is to get familiar with architecture of pointnet for classification and segmentation. Add multi-gpu support and fix some minor issues in [1]. Current version only supports shapenet, for modelnet40 dataaset needs additional handling for __getitem__.

## Usage

Download dataset shapenet from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip.

(optional) download dataset modelnet40 from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip. Possibly neet to convert .off to .ply for further processing.

Run the below commands:
```
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type shapenet --gpu 0 1 2 3 
```
```
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> --gpu 0
```

## Source
[1] Original code: https://github.com/fxia22/pointnet.pytorch.

[2] PointNet paper: https://arxiv.org/abs/1612.00593.
