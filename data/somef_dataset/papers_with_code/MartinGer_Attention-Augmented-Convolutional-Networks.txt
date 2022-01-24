# Attention-Augmented-Convolutional-Networks
This is a pytorch implementation of the paper [Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925 "Attention Augmented Convolutional Networks") by Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens and Quoc V. Le.

## Method
The paper implements the attention mechanism into different ResNet architectures. Convolutional and attentional feature maps are concatenated in a choosen ratio.

![aacn_image](https://user-images.githubusercontent.com/19909320/119885192-cd15e900-bf31-11eb-985b-be4e09ac9a4c.png)

*  ![v](https://user-images.githubusercontent.com/19909320/119885127-b40d3800-bf31-11eb-8165-1b12a739179f.png) 
is the ratio of attentional channels to number of original output filters

* ![k](https://user-images.githubusercontent.com/19909320/119885316-f9316a00-bf31-11eb-96e4-97134fc0dfb1.png) 
is the ratio of key depth to number of original output filters

![AACN](https://user-images.githubusercontent.com/19909320/137499701-4cace468-ffa5-4b2e-b15f-14d2ddee4fbf.png)

## Implementation details
I only tested the implementation with ResNet50 for now. The used ResNet V1.5 architectures are adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

The implemented memory efficient version of Relative Self-Attention is adapted with slight changes from the paper. The authors write *The relative positional embeeddings rH and rW are learned and shared across heads but not layers.* I don't think they are actually shared across heads in their implementation and changed that accordingly.

#### Additional Parameters:
- attention: ResNet stages in which you would like to apply the attention layers
- num_heads: Number of attention heads
- inference: Allows to inspect the attention weights of a trained model

## Example
See the jupyter notebook or the example training script

## Requirements
- pytorch
- I use [fast.ai](https://www.fast.ai/) and the [imagenette](https://github.com/fastai/imagenette) dataset for the examples
