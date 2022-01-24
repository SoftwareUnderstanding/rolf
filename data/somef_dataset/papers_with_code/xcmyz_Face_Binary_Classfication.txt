## Description
该工作处理的是图像的二分类问题。判断是否为人脸，做为人脸检测的一个模块。主体模型是[ResNet](https://arxiv.org/abs/1512.03385)

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [LFW Face Database](http://vis-www.cs.umass.edu/lfw/)

## Dependencies
* Python 3.6
* Pytorch 1.0.1
* Torchvision 0.2.1
* Numpy 1.16.0
* Matplotlib 2.0.2
* Cuda 9.0

## Prepare For Training Data
1. 下载LFW Face Database数据集，解压后放置于`data`下
2. 运行`data/prepare_data.py`
4. 注：在`data/prepare_data.py`第105行增加参数`download=True`，可以自动下载cifar100
3. 运行`gen_train_loader_data.py`

## Run
1. 运行`train.py`来训练模型
2. 运行`test.py`来测试

## Some Details
1. 因为本人计算机内存有限，所以将training data拆分放置于`database`下，如果你的内存足够大，则不需要这么做
2. ResNet模型优秀，收敛更快，准确度更高，感谢Kaiming He大神

## Result

![result2.jpg](loss.jpg)

## Reference
[1] Kaiming He, et al. "Deep Residual Learning for Image Recognition." arXiv arXiv:1512.03385 (2015).
