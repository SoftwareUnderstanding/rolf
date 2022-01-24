## Intro
文章《Squeeze-and-Excitation Networks(CVPR2018)》(https://arxiv.org/abs/1709.01507v4), 在神经网络中加入了注意力机制，取得了非常好的实验结果.

本code主要参考了 https://github.com/moskomule/senet.pytorch, 感谢! 本人按个人习惯进行了重写.

## Run
运行 resnet
> python train.py --dir=weight --epochs=100 --batch_size=32 --lr_init=0.001 

运行 se_resnet
> python train.py --dir=weight --epochs=100 --batch_size=32 --lr_init=0.001 --se
