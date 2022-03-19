# DCGAN

Pytorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for CelebA dataset.

A DCGAN is a direct extension of the GAN , except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. 

It was first described by Radford et. al. in the paper Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks https://arxiv.org/pdf/1511.06434.pdf. 

The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations. The input is a 3x64x64 input image and the output is a probability that the input is from the data distribution. 

The generator is comprised of convolutional-transpose layers, batch norm layers,and ReLU activations. The input is a latent vector z. z is drawn from a standard normal distribution and the output is a 3x64x64 RGB image. The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image. 

Coded by following https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## Instructions to run notbook

(1) Download dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

(2) Unzip the img_align_celeba.zip and store it in folder celeba.
     
     Structure .
              ├── celeba                 # parent directory
              ├── img_align_celeba       # folder with images inside parent directory
