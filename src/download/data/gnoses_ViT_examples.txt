# Vision transformer examples - PyTorch

This repository contains PyTorch examples of Vision transformer (https://arxiv.org/abs/2010.11929). <br>
Code is originally based on https://github.com/lucidrains/vit-pytorch.

## Dataset
 1. MNIST
 2. Dogs vs cats (https://www.kaggle.com/c/dogs-vs-cats/data)

## Install

`pip install -r requirements.txt` <br>


## MNIST

Features: Hyper parameter tuning by optuna (https://github.com/optuna/optuna)

`python vit_mnist.py`

<img src="images/plot_mnist.png" alt="Training curve accuracy" width="400" height="350">


## Dogs and Cats

Download Dogs and cats dataset and unzip into 'data' directory.

`python vit_catanddog.py`

| trial                                | dim | mlp_dim | depth | heads | accuracy |
|--------------------------------------|:---:|:-------:|:-----:|:-----:|:--------:|
| baseline (supervised ResNet50)       |  -  |    -    |   -   |   -   |   98.5%  |
| efficient_dim512                     | 512 |   512   |   16  |   16  |   76.4%  |
| huge32_embed_512                     | 512 |   512   |   32  |   16  |   80.2%  |
| huge32_aug (more data augmentation)  | 512 |   512   |   32  |   16  |   93.3%  |
| pretraining (ongoing)                |     |         |       |       |          |


<img src="images/plot_vit_catanddog.png" alt="Training and validation curve accuracy" width="800" height="350">

