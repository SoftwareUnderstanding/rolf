# cyclegan

This repo is still a work in progress. Currently, my implementation does not perform well at train time and I am still looking into reasons why this may be. I believe it is due to the learning rates of the discriminator and generator or because the loss is not being calculated correctly.

This is my current implementation of CycleGan based on Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks https://arxiv.org/pdf/1703.10593.pdf. This network is supposed to take unpaired images from two domains A and B (summer and winter, buildings and painted buildings, horses and zebras, etc) and learn a function to transform images from Domain A to Domain B, and from Domain B to Domain A. This is similar to style transfer but with the notion that you do not need paired images to do the translation (unlike the other style transfer net repo I have). CycleGans pit 2 discriminator networks and 2 generator networks against each other during training. Generator A is supposed to try to fool Discriminator B into thinking its generated images come from Domain B. Generator B is supposed to try to fool Discriminator A into thinking its generated images come from Domain A. The discriminators are trying to learn how to discriminate between the real images and the generated fake images. All 4 networks are trained at the same time. 

### Code
cyclegan.py: This file instantitates 2 instances of the generator and discriminator networks as described in the paper. It also defines loss functions, and forward and backward propagation.
disc.py: Defines the discriminator network as described by the paper
gen.py: Defines the generator network as described by the paper
run.py : This is the main file that calls the training loop for the cyclegan architecture.
