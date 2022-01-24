# Color contest submission

by: Clarence Huang
clarence.n.huang@gmail.com

## To run 

First make sure all dependencies are installed. This solution is based on pytorch and at a minimum, you need Pytorch 0.4 as well as Torchvision. To run

`python run.py trained_models/best_so_far.pth INPUT_IMAGE OUTPUT_IMAGE

Example:

`python run.py trained_models/best_so_far.pth /data/train/1055022497-hydrangea.jpg output.jpg`

## Some technical details

This solution uses inspiration from the `ResNext-152[1]` as well as `UNet[2]`. A modified  resnext is used as the base image feature extractor and the UNet segmentor is used as an upscaler. 

The approach used is classification based instead of regression based for better color reproduction. Since colors are not normally distributed, using MSE as a loss measure produces very washed out images. 

The loss function used is a custom soft encoded KL-divergence. 

## Training

The base extractor is trained on imagenet classification task. Then segmentor is then frozen and the upscaler is separately trained on the imagenet 'flowers' synthset dataset. The whole net is then unfrozen and fine tuned on the training images provided by lukas. We make use of cosine annealing as well as differential learning rates in final stages of trainng.


## References

[1][https://github.com/facebookresearch/ResNeXt] 

[2]https://arxiv.org/abs/1505.04597







