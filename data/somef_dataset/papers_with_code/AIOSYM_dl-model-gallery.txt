# Deep Learning Model Gallery
Implementation of various popular Deep Learning models and architectures. 

## LeNet
**Paper**: [Gradient-based learning applied to document recognition] (http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (1998)

**LeNet or LeNet-5** (because it has 5 layers that contain parameters) is the most basic form of Convolutional Neural Network (CNN) architecture. It was first used for handwritten digit recognition ([MNIST](http://yann.lecun.com/exdb/mnist/) dataset)

**Architecture Summary**
![LeNet Architecture](figures/lenet.jpg)

|      Layer      |  Kernel Size | Number of Filter | Stride | Padding | Activation Function |  Output Size |
|:---------------:|:------------:|:----------------:|:------:|:-------:|:-------------------:|:------------:|
|      Input      |  32 x 32 x 3 |                  |        |         |                     |              |
|   Convolution   |     5 x 5    |         6        |    1   |    0    |          -          |  28 x 28 x 6 |
|     Maxpool     |     2 x 2    |         -        |    2   |    0    |       sigmoid       |  14 x 14 x 6 |
|   Convolution   |     5 x 5    |        16        |    1   |    0    |          -          | 10 x 10 x 16 |
|     Maxpool     |    2 x 2     |         -        |    2   |    0    |       sigmoid       |  5 x 5 x 16  |
| Fully Connected |      120     |         -        |    -   |    -    |          -          |      120     |
| Fully Connected |      84      |         -        |    -   |    -    |          -          |      84      |
| Fully Connected | 10 (classes) |         -        |    -   |    -    |       softmax       |      10      |

| Framework   | PyTorch | TensorFlow |
|-------------|---------|------------|
| Implemented |   [:white_check_mark:](lenet_pytorch.py)  |   :ballot_box_with_check: |

## AlexNet

**Paper**: [ImageNet Classification with Deep Convolutional Neural Networks] (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (2012)

**AlexNet** competed in the ImageNet Large Scale Visual Recognition Challenge and achieved a top-5 error of 15.3%. This model has made the name of Deep Learning as well known as today. Also, there is a special layer called LocalResponseNorm(LRN). However, there are some experiments that proof that LRN does not contribute much. 

**Architecture Summary**

<div align="center">
<img src="figures/alexnet.png" alt="" title="AlexNet">
</div>

<a href="https://blogs.technet.microsoft.com/machinelearning/2017/04/12/embarrassingly-parallel-image-classification-using-cognitive-toolkit-tensorflow-on-azure-hdinsight-spark/"><p style="text-align:center">Image source</p></a>


Note: If you read the original paper, the figure above will be a bit different. Because the author use 2 GPUs for training and the figure illustrate distribution of each layter to those GPUs.


|       Layer       |   Kernel Size  | Number of Filter | Stride | Padding | Activation Function |  Output Size  |
|:-----------------:|:--------------:|:----------------:|:------:|:-------:|:-------------------:|:-------------:|
|       Input       |  227 x 227 x 3 |                  |        |         |                     |               |
|    Convolution    |     11 x 11    |        96        |    4   |    0    |         ReLU        |  55 x 55 x 96 |
| LocalResponseNorm |        -       |         -        |    -   |    -    |          -          |  55 x 55 x 96 |
|     MaxPooling    |     3 x 3      |         -        |    2   |    0    |          -          |  27 x 27 x 96 |
|    Convolution    |      5 x 5     |        256       |    1   |    1    |         ReLU        | 27 x 27 x 256 |
| LocalResponseNorm |        -       |         -        |    -   |    -    |          -          | 27 x 27 x 256 |
|     MaxPooling    |     3 x 3      |         -        |    2   |    0    |          -          | 13 x 13 x 256 |
|    Convolution    |      3 x 3     |        384       |    1   |    1    |         ReLU        | 13 x 13 x 384 |
|    Convolution    |      3 x 3     |        384       |    1   |    1    |         ReLU        | 13 x 13 x 384 |
|    Convolution    |      3 x 3     |        256       |    1   |    1    |         ReLU        | 13 x 13 x 256 |
|     MaxPooling    |      3 x 3     |         -        |    2   |    0    |          -          |  6 x 6 x 256  |
|  Fully Connected  |      4096      |         -        |    -   |    -    |         ReLU        |      4096     |
|  Fully Connected  |      4096      |         -        |    -   |    -    |         ReLU        |      4096     |
|  Fully Connected  | 2 (classes) |         -        |    -   |    -    |       softmax       |      2     |

| Framework   | PyTorch | TensorFlow |
|-------------|---------|------------|
| Implemented |   [:white_check_mark:](alexnet_pytorch.py)  |   :ballot_box_with_check: |

## VGG-16

**Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition] (https://arxiv.org/pdf/1409.1556.pdf) (2014)

**VGG-16** is a model proposed by the researchers at the University of Oxford. This model uses many convolution layers on top of each other to extract features. VGG-16 here derived from the fact that there are 13 convolution layers and 3 fully connected layers (13+3=16). You might also see VGG-19 as well. So refer to the paper for more detail of the architecture. 

**Architecture Summary**

![VGG Architecture](figures/vgg.png)

|      Layer      |  Kernel Size  | Number of Filter | Stride | Padding | Activation Function |   Output Size   |
|:---------------:|:-------------:|:----------------:|:------:|:-------:|:-------------------:|:---------------:|
|      Input      | 224 x 224 x 3 |                  |        |         |                     |                 |
|   Convolution   |     3 x 3     |        64        |    1   |    1    |         ReLU        |  224 x 224 x 64 |
|   Convolution   |     3 x 3     |        64        |    1   |    1    |         ReLU        |  224 x 224 x 64 |
|    MaxPooling   |     2 x 2     |         -        |    2   |    0    |          -          |  112 x 112 x 64 |
|   Convolution   |     3 x 3     |        128       |    1   |    1    |         ReLU        | 112 x 112 x 128 |
|   Convolution   |     3 x 3     |        128       |    1   |    1    |         ReLU        | 112 x 112 x 128 |
|    MaxPooling   |     2 x 2     |         -        |    2   |    0    |          -          |  56 x 56 x 128  |
|   Convolution   |     3 x 3     |        256       |    1   |    1    |         ReLU        |  56 x 56 x 256  |
|   Convolution   |     3 x 3     |        256       |    1   |    1    |         ReLU        |  56 x 56 x 256  |
|   Convolution   |     3 x 3     |        256       |    1   |    1    |         ReLU        |  56 x 56 x 256  |
|    MaxPooling   |     2 x 2     |         -        |    2   |    0    |          -          |  28 x 28 x 256  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  28 x 28 x 512  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  28 x 28 x 512  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  28 x 28 x 512  |
|    MaxPooling   |     2 x 2     |         -        |    2   |    0    |          -          |  14 x 14 x 512  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  14 x 14 x 512  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  14 x 14 x 512  |
|   Convolution   |     3 x 3     |        512       |    1   |    1    |         ReLU        |  14 x 14 x 512  |
|    MaxPooling   |     2 x 2     |         -        |    2   |    0    |          -          |   7 x 7 x 512   |
| Fully Connected |      4096     |         -        |    -   |    -    |         ReLU        |       4096      |
| Fully Connected |      4096     |         -        |    -   |    -    |         ReLU        |       4096      |
| Fully Connected |  2 (classes)  |         -        |    -   |    -    |       softmax       |        2        |

| Framework   | PyTorch | TensorFlow |
|-------------|---------|------------|
| Implemented |   [:white_check_mark:](vgg_pytorch.py)  |   :ballot_box_with_check: |


## GoogLeNet

**Paper**: [Going Deeper with Convolutions] (https://arxiv.org/abs/1409.4842) (2014)

**GoogLeNet** is a 22 layers deep network and this name is honor to
Yann LeCuns pioneering LeNet 5 network. It introduces a new module called **Inception**. 

**Architecture Summary**

![Inception Module](figures/inception.png)

![](figures/googlenet.png)

|  Layer | Kernel Size/Stride/Padding | #Filter | For inception module |  |  |  |  |  | Activation Function | output size |
| :---: | :---: | :---: | :---: | --- | --- | --- | --- | --- | :---: | :---: |
|   |  |  | #1x1 | #3x3 | #5x5 | pooling | #3x3(reduce) | #5x5(reduce) |  |  |
|  Input | 224x224x3 |  |  |  |  |  |  |  |  |  |
|  Convolution | 7x7 / 2 / 0 | 64 | - | - | - | - | - | - | ReLU | 112x112x64 |
|  MaxPool | 3x3 / 2 | - | - | - | - | - | - | - | - | 56x56x64 |
|  LocalResponseNorm | - | - | - | - | - | - | - | - | - | 56x56x64 |
|  Convolution | 1x1 / 1 / 0 | 64 | - | - | - | - | - | - | ReLU | 56x56x64 |
|  Convolution | 3x3 / 1 / 1 | 192 | - | - | - | - | - | - | ReLU | 56x56x192 |
|  LocalResponseNorm | - | - | - | - | - | - | - | - | - | 56x56x192 |
|  MaxPool | 3x3 / 2 | - | - | - | - | - | - | - | - | 28x28x192 |
|  Inception(3a) | - | - | 64 | 128 | 32 | 32 | 96 | 16 | - | 28x28x256 |
|  Inception(3b) | - | - | 128 | 192 | 96 | 64 | 128 | 32 | - | 28x28x480 |
|  MaxPool | 3x3 / 2 | - | - | - | - | - | - | - | - | 14x14x480 |
|  Inception(4a) | - | - | 192 | 208 | 48 | 64 | 96 | 16 | - | 14x14x512 |
|  Inception(4b) | - | - | 160 | 224 | 64 | 64 | 112 | 24 | - | 14x14x512 |
|  Inception(4c) | - | - | 128 | 256 | 64 | 64 | 128 | 24 | - | 14x14x512 |
|  Inception(4d) | - | - | 112 | 288 | 64 | 64 | 144 | 32 | - | 14x14x528 |
|  Inception(4e) | - | - | 256 | 320 | 128 | 128 | 160 | 32 | - | 14x14x832 |
|  MaxPool | 3x3 / 2 | - | - | - | - | - | - | - | - | 7x7x832 |
|  Inception(5a) | - | - | - | - | - | - | - | - | - | 7x7x832 |
|  Inception(5b) | - | - | - | - | - | - | - | - | - | 7x7x1024 |
|  AvgPool | 7x7 / 1 | - | - | - | - | - | - | - | - | 1x1x1024 |
|  Dropout (40%) | - | - | - | - | - | - | - | - | - | 1x1x1024 |
|  Fully Connected | 2 (classes) | - | - | - | - | - | - | - | softmax | 1x1x2 |

| Framework   | PyTorch | TensorFlow |
|-------------|---------|------------|
| Implemented |   [:white_check_mark:](googlenetv1_pytorch.py)  |   :ballot_box_with_check: |


