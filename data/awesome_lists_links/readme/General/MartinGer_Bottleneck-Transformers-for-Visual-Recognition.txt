# Bottleneck-Transformers-for-Visual-Recognition
This is a pytorch implementation of the paper [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605 "Bottleneck Transformers for Visual Recognition") by Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel and Ashish Vaswani.

## Method
This paper implements the attention mechanism into different ResNet architectures. The used design of one attention layer can be displayed as shown

![BoTNet](https://user-images.githubusercontent.com/19909320/137499600-327e82d6-f6ac-42ac-8b59-7b195caac090.png)

Global Self-Attention on images is subject to the problem, that it can only be applied after significant
spatial downsampling of the input. Every pixels relation is calculated to every other pixel so learning gets computationally very expensive, which prevents its usage across all layers in a fully attentional model.

Consequently in this paper the authors only apply their attention mechanism in the last stage of the respective architecture, here ResNet50:

![botnet](https://user-images.githubusercontent.com/19909320/119896460-567fe800-bf3f-11eb-815b-db71ae6b1908.png)

## Implementation details
I only tested the implementation with ResNet50 for now. The used ResNet V1.5 architectures are adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

The implemented memory efficient version of Relative Self-Attention is adapted with slight changes from the paper [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925 "Attention Augmented Convolutional Networks")

#### Additional Parameters:
- attention: ResNet stages in which you would like to apply the attention layers
- num_heads: Number of attention heads
- inference: Allows to inspect the attention weights of a trained model

## Example
See the jupyter notebook or the example training script

## Requirements
- pytorch
- I use [fast.ai](https://www.fast.ai/) and the [imagenette](https://github.com/fastai/imagenette) dataset for the examples
