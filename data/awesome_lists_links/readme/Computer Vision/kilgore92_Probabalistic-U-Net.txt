## PyTorch implementation of a Probabalistic U-Net

As of now, we have only implemented the basic U-Net and plan to develop utilities for data augmentation, training and inference before moving on to extending the existing U-Net model. For details about the U-Net please refer to : http://arxiv.org/abs/1505.04597

In addition to the model, we also provide a PyTorch Dataset wrapper (WIP) for the [CHAOS Liver MR dataset](https://chaos.grand-challenge.org/). 

Packages used and their versions:
* PyTorch 1.0.1
* Numpy 1.15.4
* imageio 2.5.0
* Tensorflow 1.13.1 (To use [tensorboardX](https://github.com/lanpa/tensorboardX) for viz)
* [gryds](https://github.com/tueimage/gryds) for augmenting data with deformations


This project is a WIP. The goal is to have a stable implementation of the [Probablistic U-Net](https://arxiv.org/abs/1806.05034)


#### Usage
TODO

#### Pending tasks:
* <strike>Write function to calculate dice similarity between predicted seg-map and ground truth</strike>
* <strike>Track loss on train data v/s loss on val data during training</strike> 
* <strike>Display images,predicted segmentations and ground truth for better debugging and performance analysis</strike>
*  <strike>Creation of groud truth segmentation maps from label images needs to be fixed. For correct normalization (across all classes), a background class is needed</strike> 
* <strike>Additional data augmentation using [gryds](https://github.com/tueimage/gryds)</strike>
* <strike>Hyper-parameter/architecture tuning</strike>
* Implementing Prior and Posterior nets for Prob. U-Net

Code has been tested on Python 3.7.4
