# FaceGenerationWithGAN

## Overview 
GANs is unsupervised deep-learning model that able to generate new samples from training data. This is possible because in GANs we are training two models, Discriminator and Generator. The Generator tries to generate fake inputs, and the Discriminator is like normal classifier, trying to predict which inputs are real and fake. GANs is like competitive game where each model is trying to beat each other, if the Discriminator can distinguish which inputs are real and fake, this forces the Generator to generate fake inputs to be detected as real label.

In this Udacity project, I was required to implement DCGAN to generate recognizable human face. DCGAN combines both CNNs and GANs to generate new images that have similiarity to real images. If you'd like to learn more about DCGAN <a href=https://arxiv.org/pdf/1511.06434.pdf>click here</a> 
