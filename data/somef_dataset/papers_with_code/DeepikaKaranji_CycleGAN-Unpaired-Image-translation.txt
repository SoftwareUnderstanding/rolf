# CycleGAN-Unpaired-Image-translation    
[![GitHub license](https://img.shields.io/github/license/DeepikaKaranji/CycleGAN-Unpaired-Image-translation)](https://github.com/DeepikaKaranji/CycleGAN-Unpaired-Image-translation/blob/master/LICENSE)
[![Generic badge](https://img.shields.io/badge/tensorflow-1.0.0-blue.svg)](https://shields.io/)  

Image to Image translation involves generating a new synthetic version of a given image with some modifications. Obtaining and constructing paired image datasets is  expensive and sometimes impossible. This project implements Cycle GAN to achieve image translation from Apple to Orange with Tensorflow 1.0.0 and Python3. 

## Introduction

Training a model for image-to-image translation typically requires a large dataset of paired examples.These datasets can be difficult and expensive to prepare, and in some cases impossible, such as photographs of paintings by long dead artists. The CycleGAN is a technique that involves the automatic training of image-to-image translation models without paired examples. The models are trained in an unsupervised manner using a collection of images from the source and target domain that do not need to be related in any way.

Pix2Pix GAN, a popular image to image translation uses paired images. Paired images dataset is small and hence it is hard to implement object transformation. Cycle GANs use unpaired images for training for image to image transformation.

## Architecture
 
### **Generator (G)**
Input: Takes photos from Domain X (Apple/Horse)    
Output: Generates photos of Domain Y (Orange/Zebra).  
![generator](gen.png)

The Generator layers are:
- **Encoder**   
3 Convolution Layers with Relu as the activation function.
    - 7x7 with 1 stride and 32 filters 
    - 3x3 with 2 stride and 64 filters
    - 3x3 with 2 stride 128 filters  along
- **Transformer**   
6 resnet layers (residual block) each of which consists of 2 Convolution Layers (3x3 convolution) with 128 filters and 2 strides and relu as the activation function. 
- **Decoder**
    - 2 deconvolution layers fractionally strided with stride ½ with relu activation function
        - 3x3 with 64 filters 
        - 3x3 with 32 filters 
    - 1 conv - tanh 7x7 1 stride 3 filters

### **Discriminator (Dy)**
Input: Takes photos from  Domain Y (Orange/Zebra) and output generated from F.                                         
Output: Likelihood that the image is from Domain Y.   

![discriminator](disc.png)

The Discriminator layers are:
- 4 Convolution layers 4x4 with 2 strides each with a leaky relu activation function
    - 64 filters
    - 128 filters
    - 256 filters
    - 512 filters
- 1 last convolution layer 4x4 with 1 stride and 512 filters


## Results

`CycleGAN_report.pdf` contains a detailed report on the approach, implementation, code modules and results and references. 

![results](results.png)

## Applications
CycleGAN has several important applications:

- Generating a realistic rendering of what a building would look like based on its blueprints.
- Rendering a realistic representation of how a suspect’s face would look based on police sketch.
- Creating an image of how a location would look in each different season.
- Enhancing aspects of photos to make them look more professional.
- Conversion of MRI Scans into CT Scans
- Photo generation from paintings

## References
[1]https://arxiv.org/pdf/1703.10593.pdf   
[2]https://machinelearningmastery.com/what-is-cyclegan/   
[3]https://www.youtube.com/watch?v=NyAosnNQv_U&t=553s   
[4]https://machinelearningmastery.com/what-is-cyclegan/   
