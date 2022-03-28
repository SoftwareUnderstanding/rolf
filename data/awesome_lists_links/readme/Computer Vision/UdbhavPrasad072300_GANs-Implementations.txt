# GANs-Implementations

<p>
  <a href="https://github.com/UdbhavPrasad072300/GANs-Implementations/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/UdbhavPrasad072300/GANs-Implementations">
  </a>
  <a href="https://pypi.org/project/GANs-Implementations/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/v/GANs-Implementations">
  </a>
  <a href="https://pypi.org/project/GANs-Implementations/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/GANs-Implementations">
  </a>
  <a href="https://pypi.org/project/GANs-Implementations/">
        <img alt="Package Status" src="https://img.shields.io/pypi/status/GANs-Implementations">
  </a>
</p>

GANs Implementations and other generative models + Training (in ./notebooks)

Implemented:
<ul>
    <li>Vanilla GAN</li>
    <li>DCGAN - Deep Convolutional GAN</li>
    <li>WGAN - Wasserstein GAN</li>
    <li>SNGAN - Spectrally Normalized GAN </li>
    <li>SRGAN - Super Resolution GAN </li>
    <li>StyleGAN</li>
    <li>Pix2PixHD</li>
    <li>C-VAE - Convolutional Variational Auto-encoder</li>
</ul>

## Installation

<a href="https://pypi.org/project/gans-implementations/">PyPi Installation</a>

```bash
$ pip install gans-implementations
```

Local Install and Run: 

```bash
$ cd {PROJECT_DIRECTORY}
$ pip install -e .
```

## Example

In notebooks directory there is a notebook on how to use each of these models for their intented use case; such as image 
generation for StyleGAN and others. Check them out!

```python
from gans_package.models import StyleGAN_Generator, StyleGAN_Discriminator

in_channels = 256
out_channels = 3
hidden_channels = 512
z_dim = 128
mapping_hidden_size = 256
w_dim = 512
synthesis_layers = 5
kernel_size=3

in_size = 3
d_hidden_size = 16

g = StyleGAN_Generator(in_channels, 
                       out_channels, 
                       hidden_channels, 
                       z_dim, 
                       mapping_hidden_size, 
                       w_dim, 
                       synthesis_layers, 
                       kernel_size, 
                       device=DEVICE).to(DEVICE)

d = StyleGAN_Discriminator(in_size, d_hidden_size).to(DEVICE)

import torch

noise = torch.randn(BATCH_SIZE, z_dim).to(DEVICE)

fake = g(noise)
pred = d(fake)
```

## Handwritten Digits - MNIST 

<ul>
  <li><a href="https://github.com/UdbhavPrasad072300/GANs-Implementations/blob/main/notebooks/GAN%20with%20BCE%20-%20MNIST.ipynb">GAN with BCELoss</a></li>
  <li><a href="https://github.com/UdbhavPrasad072300/GANs-Implementations/blob/main/notebooks/DCGAN%20with%20BCE%20-%20MNIST.ipynb">DCGAN with BCELoss</a></li>
  <li><a href="https://github.com/UdbhavPrasad072300/GANs-Implementations/blob/main/notebooks/SN-WGAN%20with%20GP%20-%20MNIST.ipynb">SN-WGAN with Wasserstein Loss</a></li>
</ul>

## Work Cited

https://arxiv.org/pdf/1711.11585.pdf

https://arxiv.org/pdf/1609.04802v5.pdf

https://arxiv.org/pdf/1812.04948.pdf

https://www.coursera.org/specializations/generative-adversarial-networks-gans?
