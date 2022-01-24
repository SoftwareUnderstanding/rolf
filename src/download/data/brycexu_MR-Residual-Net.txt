This repository contains my evaluations of Merge and Run in Binarized Residual Neural Work

Copyright: Xianda Xu xiandaxu@std.uestc.edu.cn

Evaluation on Cifar-10
--------

| Accuracy         | Full-Precise (1)        | Binarized (1)  | Binarized (2)  | Netscope           |
| ---------------- |:----------------------:|:--------------:|:--------------:|:------------------:|           
| ResNet-18        | 93.28%                 | 90.50%         |                | [Network](http://ethereon.github.io/netscope/#/gist/20db0c9bcdf859d2ffa0a5a55fe9b979)       |
| MR-ResNet-20     | 92.15%                 | 87.77%         | 90.48%         | [Network](http://ethereon.github.io/netscope/#/gist/46029162791a6f9b6a9e54e7742c12d4)       |
| MR-ResNet-32     | 93.39%                 | 90.46%         | 92.58%         | [Network](http://ethereon.github.io/netscope/#/gist/02d5971a830d6c71a8a96a0a65ab3016)       |  

Style 1 does downsampling by using concatenation and convolusion (kernel_size=1, stride=2, padding=0)

Style 2 does downsampling by using concatenation and average pooling (kernel_size=2, stride=2)

Binarization Principle
---------

* Keep full-precision on the first convolutional layer and the last linear layer.

* In binarized convolutional layers, all weights are binarized and scaled in propagation (https://arxiv.org/abs/1603.05279). But here, the scale factors are not learnt but all set to 1.

* BatchNorms in binarized blocks have no affine weights and bias parameters.

* Since activations are not binarized, ReLU is used instead of HardTanh (https://arxiv.org/pdf/1602.02830).

Baseline
--------
### Model: ResNet-18

Paper: (https://arxiv.org/abs/1512.03385)

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/20db0c9bcdf859d2ffa0a5a55fe9b979)

Full-Precise Accuracy on Cifar-10: 93.28% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/Base.png"/></div>

Binarized Accuracy on Cifar-10: 90.50% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/Base (binarized).png"/></div>

Merge and Run
---------
### Model: MR-ResNet-20 (the number of layers is almost identical to the baseline ResNet-18 model)

Paper: (https://arxiv.org/abs/1611.07718)

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/46029162791a6f9b6a9e54e7742c12d4)

Full-Precise Accuracy on Cifar-10: 92.15% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR.png"/></div>

Binarized Accuracy on Cifar-10: 87.77% with 80 epoches

* Downsampling is done by firstly concatenating left-branch and right-branch and secondly using a convolusion (kernel-size:1, stride:2, padding:0)

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-18 (binarized, conv).png"/></div>

Binarized Accuracy on Cifar-10: 90.48% with 80 epoches

* Downsampling is done by firstly concatenating left-branch and right-branch and secondly using a average pooling (kernel_size:2, stride:2)

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-18 (binarized, avg).png"/></div>

### Model: MR-ResNet-32 (the depth is identical to the baseline ResNet-18 model)

Paper: (https://arxiv.org/abs/1611.07718)

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/02d5971a830d6c71a8a96a0a65ab3016)

Full-Precise Accuracy on Cifar-10: 93.39% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-32.png"/></div>

Binarized Accuracy on Cifar-10: 90.46% with 80 epoches

* Downsampling is done by firstly concatenating left-branch and right-branch and secondly using a convolusion (kernel-size:1, stride:2, padding:0)

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-32 (binarized, conv).png"/></div>

Binarized Accuracy on Cifar-10: 92.58% with 80 epoches

* Downsampling is done by firstly concatenating left-branch and right-branch and secondly using a average pooling (kernel_size:2, stride:2)

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-32 (binarized, avg).png"/></div>






















