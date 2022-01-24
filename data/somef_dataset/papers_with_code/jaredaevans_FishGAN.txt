# FishGAN
Construction of a fish generator  
 
Why fish?  Many fish are very beautiful with extremes in colorations, shapes, and patterns that have a tremendous spread in presentation.  Rather than aspiring to construct a generator that would make a fish that looks obviously like a subtly different version of an existing fish, my hope is to construct a generator that can make beautiful images of fish that look both unlike any fish I have seen, yet clearly fishy.  
![WGAN-GP GIF](Notebooks/WGAN-GP-64/WGANGPU64.gif)  

## WGAN-GP-64 (current version) 
This is heavily influenced by ProgressiveGAN (below), but I did not use the progressive aspect. Overall, a big success.  I shrunk the figures to 64x64 and then used those to train the GAN.  See Notebooks/WGAN-GP-64 for specific implementation.  Here I think the critic was unable to keep up with the generator in some regards - particularly with fins and eyes, so I suspect strengthening the critic (more layers or more filters) might help. 

## WGAN-GP
Mostly as above.  I shrunk the figures to 32x32 and then used those to train the GAN.  See Notebooks/WGAN-GP for specific implementation.

## ProgressiveGAN with WGAN-GP
- Very promising, but very, very slow to train.
- Could get nice images up to 16x16, were very unstable at 32x32, and dissolved at 64x64.
- Notebook can be found in Notebooks/ProGAN.

## Simple DCGAN 
- First attempt at a GAN, went okay - mostly, all modifications made it worse.
- See Notebooks/Simple-DCGAN for code and more extensive README.

### Credit to the following sources for many excellent explanations and some code:
* https://arxiv.org/abs/1710.10196 (ProGAN)
* https://keras.io/examples/generative/wgan_gp/
* https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py
* https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/
* https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
* https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
* https://distill.pub/2016/deconv-checkerboard/
* <ins>Hands on Machine Learning with Scikit-Learn, Keras, and Tensorflow</ins> by Aurelien Geron
* <ins>Deep Learning with Python</ins> by Francois Chollet
