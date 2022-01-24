# DCGAN-landscapes [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/4ndyparr/DCGAN-landscapes/master)

Deep Convolutional **GAN** trained to generate **landscape paintings**. *(work in progress)*

![alt text](https://github.com/4ndyparr/DCGAN-landscapes/blob/master/landscapes.png)
I am still trying different training configurations (training GANs is not easy!), but this is a sample of generated images by the best model so far.

The architecture of the network is based on the PyTorch implementation from the paper **'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks'** by Soumith Chintala. https://arxiv.org/abs/1511.06434

Some important changes are:
  - **Doubled image size:** now **128x128** instead of 64x64 (adding a layer in both networks)
  - **Unbalanced G/D channels:** now **ngf=160, ndf=40** instead of ngf=64, ndf=64 (to give an advantage to the generator) 
     
![alt text](https://github.com/4ndyparr/DCGAN-landscapes/blob/master/Generator-128.png)
![alt text](https://github.com/4ndyparr/DCGAN-landscapes/blob/master/Discriminator-128.png)

The model was trained on a Jupyter Notebook, using Kaggle Kernels.

The dataset used for training was scraped from https://www.wikiart.org/ with a python scraper built specifically for the task. It includes thousands of paintings with landscapes as main theme.  

Both the training notebook and the scraper are available in this repository.



