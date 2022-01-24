# SegNet_tensorflow: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation using tensorflow.
Implements the research paper: https://arxiv.org/pdf/1511.00561.pdf 

## Prerequisites
 - Tensorflow 1.15.0
 - Numpy
 - Scipy
 - Glob
 - Numpy

## Dataset : CamVid 
 - Trainig images: 367
 - Testing images: 100
 - Validation Images: 100
 - Resolution: 360 x 480
# SegNet architecture 
There are no fully connected layers and hence it is only convolutional. A decoder upsamples its
input using the transferred pool indices from its encoder to produce a sparse feature map(s). It then performs convolution with a trainable filter bank
to densify the feature map. The final decoder output feature maps are fed to a soft-max classifier for pixel-wise classification

<img src=/segnet.png >

# Python files
 - main.py: contains the SegNet model
 - segnet.py: contains the required functions for the loss, initialization and predictions function

# Results
| Optimizers | sky | Building | Pole | Road | Pavement | Tree | Sign | Fence | Car | Pedestrian | Bicycle | GA | mIoU |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Adam+MFB  | 94.249 | 68.725 | 18.690 | 67.270 | 85.460 | 80.800 | 74.480 | 0.0062 | 90.62 | 20.866 | 15.430 | 71.249 | 38.069 | 
| Adam | 77.710 | 68.500 | 0.9844 | 89.138 | 46.100 | 85.190 | 2.50 | 0.390 | 21.11 | 0.035 | 1.790 | 68.38 | 25 |
| SGD+MFB | 73.750 | 58.238 | 11.090 | 69.270 | 69.780 | 85.930 | 74.110 | 0.0019 | 87.45 | 31.75 | 27.28 | 65.91 | 37.29
| SGD | 89.97 | 66.09 | 3.13 | 67.18 | 75.78 | 64.8 | 45.419 | 0.000 | 88 | 16.880 | 39.10 | 66.159 | 28.68 |

<img src=/image.png >
