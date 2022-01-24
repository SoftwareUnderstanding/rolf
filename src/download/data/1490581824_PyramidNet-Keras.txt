# PyramidNet-Keras
Keras implementation of PyramidNet，from paper "Deep Pyramidal Residual Networks", CVPR 2017 .  
Arxiv: https://arxiv.org/abs/1610.02915.
### base codes are from https://github.com/kazu41/ResNet and check the error.

## 一、修改参考代码错误的部分
1）参考代码中使用的是（a）original pre-activation ResNets block,修改成论文中使用的（d）block


2）参考代码中使用bottleneck block构建的结构有误，原论文中使用bottleneck block时输入输出的channel是1:4,而参考代码依旧是使用1:1，导致最后输出channel不对。本代码已进行修改。



## 二、使用原论文中的实验参数进行训练，包括：
1）设置SGD优化器：
Our PyramidNets are trained using backpropagation [15] by Stochastic Gradient Descent (SGD) with Nesterov momentum for 300 epochs on CIFAR-10 and CIFAR-100 datasets. 

2）设置学习率衰减控制
The initial learning rate is set to 0.1 for CIFAR-10 and 0.5 for CIFAR-100, and is decayed by a factor of 0.1 at 150 and 225 epochs, respectively. 

3）参数初始化、权重衰减（正则化）
The fifilter parameters are initialized by “msra” [6]. We use a weight decay of 0.0001, a dampening of 0, a momentum of 0.9

4）论文中用到的数据增广
horizontal flflipping，translation  by 4 pixels are adopted in our experiments, following the 
common practice [18].


## 三、在cifar10上进行训练，获得keras的权重
