# Generative-Adversarial-Networks
This repository is the result of a project I did on Generative Adversarial Networks for my Master's Computer Vision & Object Recognition class . The aim of the project was to study in depth Cycle consistent Generative Adversarial Networks. 

This work was mainly based on the original article on Cycle consistent GANs : https://arxiv.org/pdf/1703.10593.pdf. First, we 
reproduced some of the results presented in the article, transfering photos to Monet paintings and vice-versa, also trying out 
the algorithm on some holiday pictures. An experiment was also carried out on a new dataset, trying to perform style transfer from day to night using photos of roads taken from a car. 

Finally, using the paper on Wasserstein loss which can be found here https://arxiv.org/pdf/1701.07875.pdf, we modified the loss function used in the Cycle GAN paper to implement the Wasserstein loss and compare results with the previous loss. 

We used the Pytorch implementation of Cycle-GANS which can be found here : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, and only modified the file models.py in order to add a WGAN model which we implemented in the cycle_wgan_model.py file. Those two contributions can be found in this repository. 
