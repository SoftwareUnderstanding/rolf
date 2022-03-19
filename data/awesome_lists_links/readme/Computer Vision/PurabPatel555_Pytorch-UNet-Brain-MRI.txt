Pytorch-UNet-Brain-MRI
=======================

This is a UNet implementation in PyTorch using a modified version of the original UNet from the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" (see Credits). Notable modifications to the original implementation are: usage of "same" padding rather than no padding, usage of batch normalization, a different input image size. The training data consists of brain MRI images and masks from a Kaggle MRI segmentation dataset (see Credits). Early stopping is used in this project with a patience of 1 for demonstration purposes, but can be changed to a more optimal value. 

Usage
=====

This is a self contained Kaggle script-there are no requirements for your local environment

If you are using a Kaggle notebook with the dataset in the Credits section below, the default folder structure should already be compatible with this script. However, you can change the PATH variable in the script if you would like the model weights to be saved at a different location than the default. GPU training is recommended for timing reasons. 

Note that due to the difficulty of incorporating "same" padding in PyTorch (relative to Keras for example), the paddings are set manually based on the image size, kernel sizes, strides, etc. Therefore, if you choose to use an input size other than 256x256, you might need to manually change the paddings to ensure the "same" convolutions are preserved.  


Credits
=======

* Original UNet Paper: https://arxiv.org/abs/1505.04597
* Kaggle MRI Dataset: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
* Some of the initial data visualization code was adapted from this notebook of a Keras UNet: https://www.kaggle.com/monkira/brain-mri-segmentation-using-unet-keras/notebook
