# PointNet.mxnet
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in mxnet. The model is in `pointnet.py` or `models/`.
The implementation of this repo refers to this [code](https://github.com/fxia22/pointnet.pytorch)

# Download data and running

```
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
python train_classification.py #train 3D model classification
python train_segmentation.py # train 3D model segmentaion

```

# Performance
Without heavy tuning, PointNet can achieve 80-90% performance in classification and segmentaion on this [dataset](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html). 

Sample segmentation result:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)


# Links

- [Pytorch implementation](https://github.com/fxia22/pointnet.pytorch)
- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)

