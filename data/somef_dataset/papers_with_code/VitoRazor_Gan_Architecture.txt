# Gan_Architecture
create an architecture for Generative Adversarial Networks.
implementations 
## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Spectral Normalization](#GAN-SN)  
    + [GAN with Info](#GAN-info)

## Installation
    $ git clone https://github.com/VitoRazor/Gan_Architecture.git
    $ cd Gan_Architecture-master/
    $ pip install keras

## Implementations   
### GAN-SN
Implementation of Generative Adversarial Network with Spectral Normalization for Wasserstein-divergence 

[Code](myGan_w_sn.py)

Reference Paper:

Spectral normalization for generative adversarial networks:https://arxiv.org/abs/1802.05957

Wasserstein GAN: https://arxiv.org/abs/1701.07875

Result:
Train fro cartoon characters 64x64 
 <p align="center">
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan/example_100000.png" width="650"\>
</p>
Train fro aerial image 64x64[iteration=150000] and 256x256[iteration=34800]
 <p align="center">
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan/example_150000.png" width="400"\>
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan/example_34800.png" width="400"\>
</p>

### GAN-info
Implementation of Generative Adversarial Network with InfoGAN and ACGAN, simultaneously using Spectral Normalization for Wasserstein-divergence.

[Code](myGan_info.py)

Reference Paper:

Auxiliary Classifier Generative Adversarial Network: https://arxiv.org/abs/1610.09585

Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets: https://arxiv.org/abs/1606.03657                                             
                 
Result:
from iteration 10 to iteration 15000
<p align="left">
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan_info/example_100.png" width="400"\>
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan_info/example_10000.png" width="400"\>
</p>
<p align="center">
    <img src="https://github.com/VitoRazor/Gan_Architecture/blob/master/result/Gan_info/example_15000.png" width="400"\>
</p>

