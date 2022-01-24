# Artistic-Style-Transfer
This project demonstrates the use of Style Transfer in python, iOS, Android mobile applications inspired by [Neural Style Transfer algorithm by Gatys et al.(2015)](https://arxiv.org/abs/1508.06576).

## Project Goal 
- Implement the neural style transfer algorithm
- Generate novel artistic images using algorithm

## Background
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), such as an artwork by a famous painter or a texture photo to resemble and blend them together , in order to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S. So the output image looks like the content image, but "painted" in the style of the style reference image.
![example](style_tx_cat.png)

### Structure 
As for pre-trained convolutional model, we use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet competition database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).
VGG-19 network architecture looks as follows: 
![VGG-19](vgg19_convlayers.png)
Main folder contains two notebooks: one implemented using GPU/PyTorch and the other implemented using GPU/Tensorflow. 
* [PyTorch VGG-19 pretrained model based style transfer](Style_Transfer_PyTorch.ipynb)
* [Tensorflow VGG-19 pretrained model based style transfer](Art%2BGeneration%2Bwith%2BNeural%2BStyle%2BTransfer%2B-%2Bv2.ipynb)
    * [Pre-trianed ImageNet VGG-19 model](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
<!-- TODO --> 
These mobile platform requires lite-weight cpu-intensive model so that a pre-trained TensorFlow Lite model and its API are used. 
* Android style Transfer(/android/README.md)
* iOS Style Transfer

### Implementation Approach
- Check Cuda and Allocate Device (PyTorch/Cuda) /Create an Interactive Session(Tensorflow) 
- Load the content image
- Load the style image
- Process the content/style images 
- Load the VGG16 model and un on Cuda / Tensorflow session
- Train/Run Model on GPU:
  - Run the content image through the VGG16 model and compute the content cost
  - Run the style image through the VGG16 model and compute the style cost
  - Compute the total cost
  - Define the optimizer and the learning rate
  - tensorflow case : Build the TensorFlow graph
- Generate "combined" image

### Reference 
The Neural Style Transfer algorithm was due to Gatys et al. (2015). 
* Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015) [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
* TensorFlow Implementation of ["A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
* Harish Narayanan, [Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
* [Tensorflow Artistic-Style-Transfer](https://www.tensorflow.org/lite/models/style_transfer/overview)
* [Pytorch transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/)
