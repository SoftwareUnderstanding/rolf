<h1>
  Adversarial Latent Autoencoders, ALAE with TF2
  <br>
</h1>
  <p>
    Frédéric TOST    
  </p>
<h4>MNIST dataset, ConvNet implementation with Tensorflow 2</h4>

Generated images

![](static/alae_samples_40300_redim.png)&nbsp;&nbsp;&nbsp;
![](static/alae_static_samples_reconst.gif)

# ALAE

## Content

This is a Python/Tensorflow 2.0 implementation of the **A**dversarial **L**atent **A**uto**E**ncoders. 
See reference below: 
* Stanislav Pidhorskyi, Donald A. Adjeroh, and Gianfranco Doretto. Adversarial Latent Autoencoders. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. [to appear] 
>

<h5>preprint on arXiv: <a href="https://arxiv.org/abs/2004.04467">2004.04467</a></h5>

**MNIST** dataset is used as a toy example. The **Generator** and **E encoder** are using **Conv2D** and **Conv2DTranspose** instead of a **MLP** (Multi-Layer Perceptron used in the paper). 
This gives better results but  a longer training.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hparams.svg?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## Objective

The objective is to show how to easily implement the ALAE using the **MNIST** dataset and convolutional networks. Finding the hyperparameters such as learning rate of each optimizer is the most fastidious task. 
 
Additional features
- The use of Tensorflow 2.0 **HParams Dashboard** features allows to keep a trace of each run using different hyperparameters.

![](static/hparam_table_view.png) 


- 3 losses time history 

| Loss | Time history
| :--- | :----------
| Generator | ![](static/loss_generator.png) 
| Discriminator | ![](static/loss_discriminator.png)
| Latent | ![](static/loss_latent.png) 

- the latent loss was splitted into **Reconstruction** loss and **Kullback Leibler (KL)** loss. 
KL loss is not used in the original paper, it seems to accelerate the convergence of the Generator/Discriminator.

    
    K_RECONST_KL = 1.0 # To use full reconstruction (alae_tf2.py)

## To run the demo

To run the demo, you will need to have installed Tensorflow 2.0.0 or more recent (2.1.0, 2.2.0). 

Run the demo

    python alae_tf2.py



#### Repository structure

| Path | Description
| :--- | :----------
| alae_tf2.py | Configure the hyperparameters and run the demo.
| alae_tf2_helper.py | Train the neural network. alae_helper is the class that define the losses and the train step function.
| alae_tf2_models.py | Models used in the demo, Encoders F & E, Generator and Discriminator (4 classes).
| utils.py | Useful functions to process and plot images (2 functions).

## Authors

- Frédéric TOST — Developer [TOST Corp.](https://tostcorp.com/)