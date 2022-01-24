# Image Super-Resolution (ISR)

<img src="figures/butterfly.png">

[![Build Status](https://travis-ci.org/idealo/image-super-resolution.svg?branch=master)](https://travis-ci.org/idealo/image-super-resolution)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://idealo.github.io/image-super-resolution/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://github.com/idealo/image-super-resolution/blob/master/LICENSE)

The goal of this project is to upscale and improve the quality of low resolution images.

This project contains Keras implementations of different Residual Dense Networks for Single Image Super-Resolution (ISR) as well as scripts to train these networks using content and adversarial loss components.  

The implemented networks include:

- The super-scaling Residual Dense Network described in [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797) (Zhang et al. 2018)
- The super-scaling Residual in Residual Dense Network described in [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) (Wang et al. 2018)
- A multi-output version of the Keras VGG19 network for deep features extraction used in the perceptual loss
- A custom discriminator network based on the one described in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGANS, Ledig et al. 2017)

Read the full documentation at: [https://idealo.github.io/image-super-resolution/](https://idealo.github.io/image-super-resolution/).

[Docker scripts](https://idealo.github.io/image-super-resolution/tutorials/docker/) and [Google Colab notebooks](https://github.com/idealo/image-super-resolution/tree/master/notebooks) are available to carry training and prediction. Also, we provide scripts to facilitate training on the cloud with AWS and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with only a few commands.

ISR is compatible with Python 3.6 and is distributed under the Apache 2.0 license. We welcome any kind of contribution. If you wish to contribute, please see the [Contribute](#contribute) section.

## Contents
- [Pre-trained networks](#pre-trained-networks)
- [Installation](#installation)
- [Usage](#usage)
- [Additional Information](#additional-information)
- [Contribute](#contribute)
- [Citation](#citation)
- [Maintainers](#maintainers)
- [License](#copyright)

## Pre-trained networks

The weights used to produced these images are available directly when creating the model object. 

Currently 4 models are available:
  - RDN: psnr-large, psnr-small, noise-cancel
  - RRDN: gans
 
Example usage:

```
model = RRDN(weights='gans')
```
  
The network parameters will be automatically chosen.
(see [Additional Information](#additional-information)).

#### Basic model
RDN model, PSNR driven, choose the option ```weights='psnr-large'``` or ```weights='psnr-small'``` when creating a RDN model.

|![butterfly-sample](figures/butterfly_comparison_SR_baseline.png)|
|:--:|
| Low resolution image (left), ISR output (center), bicubic scaling (right). Click to zoom. |
#### GANS model
RRDN model, trained with Adversarial and VGG features losses, choose the option ```weights='gans'``` when creating a RRDN model.

|![baboon-comparison](figures/baboon-compare.png)|
|:--:|
| RRDN GANS model (left), bicubic upscaling (right). |
-> [more detailed comparison](http://www.framecompare.com/screenshotcomparison/PGZPNNNX)

source: [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
