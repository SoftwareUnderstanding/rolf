# ResNeSt
Split-Attention Network, A New ResNet Variant. It significantly boosts the performance of downstream models such as Mask R-CNN, Cascade R-CNN and DeepLabV3.

## Reference

**ResNeSt: Split-Attention Networks** [[arXiv](https://arxiv.org/pdf/2004.08955.pdf)]

Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```
**Cutmix: Regularization Strategy to Train Strong Classifiers with Localizable Features**
https://github.com/clovaai/CutMix-PyTorch

## environment requirement
* apex
```bash=
git clone https://github.com/NVIDIA/apex
cd apex/
python3 setup.py install --cuda_ext --cpp_ext
```
* DALI
* pytorch.1.5.0+cu101

## Prepare Data
```bash=
-train
--class1
--class2
--...
-val
--class1
--class2
--...
```

## How to Train
```bash=
##use multi gpu training
python3 -m torch.distributed.launch --nproc_per_node=4 train.py 
```


