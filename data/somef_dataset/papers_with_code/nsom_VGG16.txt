# VGG16
My Pytorch Implementation of VGG16 based off of https://arxiv.org/pdf/1409.1556.pdf

Here is a sample of the classification on CIFAR-10 after about 20 epochs of training:

![Alt text](VGG16/sample.png?raw=true)

Classifications:

Horse, Dog, Ship, Plane
Ship, Deer, Horse, Plane
Cat, Horse, Cat, Ship
Dog, Dog, Car, Horse


# Resnet34
My Pytorch Implementation of Resnet34 from https://arxiv.org/pdf/1512.03385.pdf. This model performs slightly better than VGG16, but with less training time and takes about half as much to classify a new example.

Here is a sample of the classification on CIFAR-10 after about 7 epochs of training:

![Alt text](resnet34/resnet34_test.png?raw=true)

Classifications:

Dog, Car, Ship, Plane,
Frog, Frog, Car, Frog,
Cat, Car, Plane, Truck,
Dog, Horse, Truck, Ship

# GAN

My Pytorch implementation of a Generative Adverserial Network is based on https://arxiv.org/pdf/1406.2661.pdf

Below is an example of 25 randonly selected examples from the generator over 15 epochs of training on the MNIST dataset:

![Alt text](GAN/gan_train.gif?raw=true)

# Autoencoder

Below is an example of 25 randonly selected examples from the generator over 15 epochs of training on the MNIST dataset, also included are the encoding layer which was chosen to be of only size 4:

![Alt text](autoencoder/autoencoder.gif?raw=true)
![Alt text](autoencoder/encode_layer.gif?raw=true)


# Logistic Regression

Here is an example of the weights from the logistic and softmax regressions on MNIST, SGD was used to fit them and there were 10000 samples chosen.

Logistic Regression (0/1):

![Alt text](logistic_regression/lr_theta_img.png?raw=true)

Softmax Regression:

| | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:| :-------------------------:|:-------------------------:|
| ![Alt text](logistic_regression/theta_softmax_img_0.png?raw=true)  0 |  ![Alt text](logistic_regression/theta_softmax_img_1.png?raw=true)  1| ![Alt text](logistic_regression/theta_softmax_img_2.png?raw=true)  2| ![Alt text](logistic_regression/theta_softmax_img_3.png?raw=true)  3  |  ![Alt text](logistic_regression/theta_softmax_img_4.png?raw=true)  4 |
| ![Alt text](logistic_regression/theta_softmax_img_5.png?raw=true)  5 |![Alt text](logistic_regression/theta_softmax_img_6.png?raw=true)  6  | ![Alt text](logistic_regression/theta_softmax_img_7.png?raw=true)  7| ![Alt text](logistic_regression/theta_softmax_img_8.png?raw=true)  8| ![Alt text](logistic_regression/theta_softmax_img_9.png?raw=true)  9|