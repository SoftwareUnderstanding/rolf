# MobileNetV2_pytorch_cifar
This is a complete implementation of MobileNetv2 in PyTorch which can be trained on CIFAR10, CIFAR100 or your own dataset.
This network comes from the paper below
>Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation
https://arxiv.org/abs/1801.04381v3

In this network, an inverted residual structure and depthwise convolution is used. Please refer to that paper for more detailed information

## Usage
This project is compiled and run on Python 2.7 and PyTorch 0.4.0
Here are some necessaries dependencies:
```
torch 0.4.0
torchvision 0.2.1
numpy 1.14.3
tensorboardX 1.2
```
use pip to install them first

## Train and Test
1. Download CIFAR10 or CIFAR100 dataset or prepare your own dataset like a dataloader defined in PyTorch
2. Modify ```config.py``` into your own configuration, eg. change ```image_size``` or something
3. Run ``` python main.py --dataset cifar10 --root /your/path/to/dataset/ ```

Tensorboard is also available, just use 
```bash
tensorboard --logdir=./exp_dir/summaries
```
logdir is also changable in ```config.py```

>I compared my implementation with the implementation of MG2033, for he has many stars, is quite confusing that my implementation on cifar100 has about 8% higher accuracy-74% vs 66%, my training policy is the same with his.

my implementation accuracy on cifar100:

![my_implementation](https://github.com/zym1119/MobileNetV2_pytorch_cifar/blob/master/img/mobilenetv2_test_cifar100_mine.png)

