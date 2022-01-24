# CactiNet
![Cactinet](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/Logo.png)

Google recently published both a very exciting paper and source code for a newly designed CNN (convolutional neural network) called EfficientNet, that set new records for both accuracy and computational efficiency.

[1] Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. Arxiv link: https://arxiv.org/abs/1905.11946.

CactiNet in a CNN modeled after EfficentNet with changes in no of 'mobile inverted bottleneck convolutional' layers and parameter values to identify aerial images of columnar cactus. 

## EfficientNet

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

EfficientNets were developed based on AutoML and Compound Scaling. In particular, first use AutoML Mobile framework to develop a mobile-size baseline network, named as EfficientNet-B0; Then, use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

![Effnet](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/params.png)

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency.

## Summary of the paper

* They learned that CNN’s must be scaled up in multiple dimensions. Scaling CNN’s only in one direction (eg depth only) will result in rapidly deteriorating gains relative to the computational increase needed.
As shown in the image below.

![them_gains](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/gains.jpeg)

* In order to scale up efficiently, all dimensions of depth, width and resolution have to be scaled together, and there is an optimal balance for each dimension relative to the others. 

## Data used
Data used to build the model were aerial images of cactus obtained by kaggle.

![yes](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/datayes.png)

![no](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/datano.png)


## CactiNet modeling

For the complete notebook related to modeling CactiNet check https://github.com/PotatoSpudowski/CactiNet/blob/master/Building_EfficientNet_model_using_Pytorch.ipynb

## Visualizing CactiNet weights

Visualizing the weights of the first conv layer

![conv](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/convweight.png)

### Saliency Map

The idea proposed by https://arxiv.org/abs/1312.6034 is to back-prop the output of the network with respect to a target class until the input and plot the computed gradient. This will highligh the part of the image responsible for that class.

![sal](https://github.com/PotatoSpudowski/CactiNet/blob/master/Images/sal.png)


## Using Transfer learning on EfficientNet B1, B2 and B3.

Fitting data on Google's EfficientNet models
https://github.com/PotatoSpudowski/CactiNet/blob/master/Aerial_Cactus_Identification_using_transfer_learning.ipynb

## Important Links

* EfficientNet Paper: https://arxiv.org/abs/1905.11946
* MnasNet: https://arxiv.org/pdf/1807.11626.pdf
* Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps 
https://arxiv.org/abs/1312.6034
* Kaggle data: https://www.kaggle.com/c/aerial-cactus-identification

