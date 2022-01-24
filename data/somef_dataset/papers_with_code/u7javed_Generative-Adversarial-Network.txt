# Generative-Adversarial-Network

Paper: https://arxiv.org/pdf/1406.2661.pdf

An implementation and script to training a Vanilla Generative Adversarial Network described in the paper by Ian Goodfellow. A GAN comprises of two adversarial models that play a minimax game. The Discriminator model tries to distinguish fake data from real data while the Generator tries to fool the Discriminator into classifying fake data as real. The ideal equilibrium for a GAN model is called the NASH equilibrium. 

The Discriminator Loss is defined as max([log(D(x)) + log(1 - D(G(z)))]) where D, G, x, z, G(z) are Discriminator, Generator, real data, latent vector, fake data respectively.

The Generator loss is defined as min([log(1 - D(G(z))])

There are 2 python files:
  - models.py
    - Models.py contains the architecture for both the generator and discriminator. The vanilla GAN is consists of feed-forward generator and discriminator utilizing a leaky relu as its activation function. Tanh activation is used as the output activation for the generator while a sigmoid classification activation function is used for the discriminator.
    
  - train.py (EXECTUABLE SCRIPT)
    - train.py is an executable python script taking in various hyperparameters for training as arguments. This script loads the models and trains the models through a certain number of epochs.
    - The MNIST dataset is also automatically downloaded to a chosen directory.
    - Saves sample images from each epoch to specified directory
    - Saves models at each epoch to specified directory
    
Fake data progression through epoch trains:
![](data/saved_images/epoch_progression.gif)
    
