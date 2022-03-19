# Overview
This is just a example to train CIFAR-10 by small Residual Network (ResNet) for learning TensorFlow 2.0.

# Description
* This is an assignment of [Deep Learning basic class](https://deeplearning.jp/lectures/dlb2018/) arranged a little. 
* Training CIFAR-10 by small ResNet on Google Colaboratory with TensorFlow 2.0 Alpha.
* Data is augmented by ImageDataGenerator of Keras.

# Dataset
This dataset can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

# Model
ResNet-56 from https://arxiv.org/abs/1512.03385. If you want to change a number of layer, please change a value of 'n' in the code. The default value is 9 and it makes 56 layers.

# Accuracy  
93.58% after training in 182 epochs with 2 hours and 14 minutes.  
![image](https://user-images.githubusercontent.com/40084422/59158776-98b58200-8afa-11e9-9835-f5ca35d138de.png)

# How to open it on Google Colaboratory
Please take a look at [How-to-open-ipynb-on-Google-Colaboratory](https://github.com/shoji9x9/How-to-open-ipynb-on-Google-Colaboratory).

# Reference
* https://github.com/yu4u/cutout-random-erasing
* https://github.com/jerett/Keras-CIFAR10
