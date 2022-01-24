# U-Net Implementation in <img src="./images/Pytorch_logo.png" alt="Kitten" title="A cute kitten" width="150" height="30" />
## Overview

### Dataset

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
### Model

![alt text](./images/unet_arch.png "Unet Architecture")

## Experiments
Training U-Net for 30 epoch with batch-size of 2 and input size of 3x960x720[]

* Pretrained model for 3x960x720 input image size.

## How to use
* Set hyper-parameters in train.sh for training and in eval.sh for evaluating
* run `bash train.sh` or `bash eval.sh`
* for more help, run `python train.py --help` or `python eval.py --help`

## Refrences

* O. Ronneberger, P. Fischer, and T. Brox, [*"U-Net: Convolutional Networks for Biomedical Image Segmentation"*](https://arxiv.org/abs/1505.04597), May 2015, arXiv:1505.04597