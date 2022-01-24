# gan_project
A generative adversarial network applied to high energy particle simulation.

The primary goal of this project is to faithfully reproduce high energy particle interaction images using a generative adversarial network model. Secondary goals include better informed Monte-Carlo algorithm construction based the latent representation of a fully trained GAN, as well as less expensive production of neural network training datasets for future projects.

### Project List
#### gan_basics
Basic implementation of a GAN using MNIST dataset. This small project is meant to be a proof of concept, as well as act as the first building block in the research project.

#### dcgan 
Implementation of deep convolutional GAN framework. Architecture, initializations and training parameters taken from: Radford, Goodfellow, et al 
- [arXiv: 1511.06434v2: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434v2.pdf)
- [arXiv: 1606.03498v1: Improved Techniques for Trainings GANs](https://arxiv.org/pdf/1606.03498.pdf) 

#### larcv_dcgan
Extension of above dcgan framework designed to work with single channel larcv images that have been converted to PNG format.

