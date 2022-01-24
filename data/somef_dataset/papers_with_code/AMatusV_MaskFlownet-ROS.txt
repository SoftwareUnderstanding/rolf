# MaskFlownet with ROS (Python 2.7)

By Antonio Matus-Vargas

Based on the work by Shengyu Zhao, Yilun Sheng, Yue Dong, Eric I-Chao Chang, Yan Xu.

[[arXiv]](https://arxiv.org/pdf/2003.10955.pdf) [[ResearchGate]](https://www.researchgate.net/publication/340115724)

```
@inproceedings{zhao2020maskflownet,
  author = {Zhao, Shengyu and Sheng, Yilun and Dong, Yue and Chang, Eric I-Chao and Xu, Yan},
  title = {MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

## Introduction

This repository includes:

- Training and inferring scripts using Python and MXNet
- Pretrained models of *MaskFlownet-S* and *MaskFlownet*
- ROS node for optical flow prediction with MaskFlownet

Code has been tested with Python 2.7.12 and MXNet 1.5.

## Pretrained Models

Original pretrained models are given (see `./weights/`).

## Inferring

The following script is for running the MaskFlownet node:

`python mfn_node.py CONFIG [-g GPU_DEVICES] [-c CHECKPOINT] [--resize INFERENCE_RESIZE]`

where `CONFIG` specifies the network configuration (`MaskFlownet_S.yaml` or `MaskFlownet.yaml`); `GPU_DEVICES` specifies the GPU IDs to use, split by commas with multi-GPU support; `CHECKPOINT` specifies the checkpoint to do inference on; `INFERENCE_RESIZE` specifies the resize used to do inference.

For example,

- to do prediction with *MaskFlownet* on checkpoint `000Mar17`, run `python mfn_node.py MaskFlownet.yaml -g 0 -c 000Mar17` (the output will be under `/flow`).
