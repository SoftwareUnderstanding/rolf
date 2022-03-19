# GANs
Keras Implementation of Generative Adverserial Networks

## Table of Contents
  * [Implementations](#implementations)
    + [Deep Convolutional GAN](#dcgan)
    + [GAN](#vgan)

## Implementations   
### DC-GAN
Implementation of _Deep Convolutional Generative Adversarial Network_.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1610.09585

#### Example
```
$ cd dcgan/
$ python3 dcgan.py
```

### GAN
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.

[Code](vgan/vgan.py)

Paper: https://arxiv.org/abs/1406.2661

#### Example
```
$ cd vgan/
$ python3 vgan.py
```