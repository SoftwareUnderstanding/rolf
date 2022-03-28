# Sugar, Flower, Fish or Gravel 

![Logo](https://raw.githubusercontent.com/raspstephan/sugar-flower-fish-or-gravel/master/logo/sffg-small.png)

Welcome to the repository for the [Zooniverse cloud classification project](https://www.zooniverse.org/projects/raspstephan/sugar-flower-fish-or-gravel)

*Attention: This repository still needs some cleaning up, so a few things might not work.*

## Installation

We highly recommend using Anaconda to manage your Python packages.

### Create a new conda environent
To start from scratch, create a new conda environment with the required packages by typing
```
conda create -n my-new-environment --file requirements.txt
python setup.py install
```


### From an existing conda environment
If you already have a conda environment and simply want to make sure you have all the necessary packages installed for this repository, do:

```
conda install --file requirements.txt
python setup.py install
```

### Keras GPU install

To use the GPU with Keras which is strongly recommended if you want to train a neural network (see below), you have to replace `keras` with `keras-gpu` in the requirements.txt file. Unfortunately it doesn't always work right out of the box depending on your system.

If you only want to use the pretrained models, you can also do this on CPU.

### Development install
If you want to modify the functions inside `pyclouds` you can do a development install by typing
```
python setup.py develop
```

## Download NASA Worldview Images

We are using satellite images provided by [NASA Worldview](https://worldview.earthdata.nasa.gov/). Go to the [image_download](https://github.com/raspstephan/sugar-flower-fish-or-gravel/tree/master/image_download) directory for further instructions.


## Deep learning algorithms

Right now I am using two deep learning algorithms: 1) A [Resnet](https://arxiv.org/abs/1708.02002) for object detection and 2) a [Unet](https://arxiv.org/abs/1505.04597) version that uses the [fastai](https://docs.fast.ai/) library for image segmentation.

Pretrained network versions are available here: https://doi.org/10.5281/zenodo.2565146

### Object detection with keras-retinanet

An object detection algorithm draws bounding boxes around objects of interest. In that way it does exactly what the human labelers did.

Here, we are using keras-retinanet, an implementation of a modern detection network in Keras: https://github.com/fizyr/keras-retinanet

In addition to the packages in the requirements.txt file you will need to install the following:

```
pip install keras-resnet

git clone git@github.com:fizyr/keras-retinanet.git
cd keras-retinanet
python setup.py install
```

Then follow the instructions in the `ml-retinanet` notebook for training and inference.

### Image segmentation using fastai

**WARNING**: fastai requires Python 3.7. I used a different conda environment for the fastai experiments. Follow the instructions on the fastai documentation.

In image segmentation, every pixel is assigned one category. For this we first need to create "masks", i.e. an image for every classification that indicates with category each pixel belongs to. This happens in the `create-segmentation-masks` notebook.

Then we are using the fastai library to create a modern Unet version in `ml-fastai-segmentation`.

