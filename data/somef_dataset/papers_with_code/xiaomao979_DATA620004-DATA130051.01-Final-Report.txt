# Introduction
DATA130051.01——计算机视觉 和 DATA620004——神经网络和深度学习 期末project
对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-10或CIFAR-100图像分类任务中的性能表现。  

mixup: https://arxiv.org/pdf/1710.09412v2.pdf 

cutmix: https://arxiv.org/pdf/1905.04899.pdf

cutout: https://arxiv.org/pdf/1708.04552.pdf 

Note: 首先构建baseline方法(如AlexNet, ResNet-18)，在baseline的基础上分别加入上述不同的data augmentation方法。
注意：Baseline方法也需要对比

# Code
baseline
```
python main.py
```
mixup
```
python main_mixup.py
```
cutmix
```
python main_CutMix.py
```
cutout
```
python main_Cutout.py
```
