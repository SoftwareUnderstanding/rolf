<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [1.Problem Statement](#1problem-statement)
- [2.Pipeline](#2pipeline)
  - [2.1 Data preprocessing](#21-data-preprocessing)
  - [2.2 Model building and evaluation](#22-model-building-and-evaluation)
    - [2.2.1 Logistic regression](#221-logistic-regression)
    - [2.2.2 Convolutional Neural Networks](#222-convolutional-neural-networks)
    - [2.2.3 Gated Convolutional Neural Networks](#223-gated-convolutional-neural-networks)
    - [2.2.4 Recurrent Neural Networks](#224-recurrent-neural-networks)
    - [2.2.5 VGG](#225-vgg)
- [3. Conclusion](#3-conclusion)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1.Problem Statement
The task is image classification for the modified MNIST dataset which contains 50000 examples for training and 10000 for testing. The task mainly aims at using basic models such as logistic regression, vanilla convolutional neural networks, recurrent neural networks on the dataset, as well as implementing a gated convolutional neural networks model.  

# 2.Pipeline
## 2.1 Data preprocessing 
Including uploading data from dropbox to google colab;  
splitting dataset into training and testing subsets;   
the digits (x) are size-normalized and centered in a fixed-size image(64*64 pixels) with values from 0 to 255;  
the labels (y) are one hot encoded to feed the models.

## 2.2 Model building and evaluation
### 2.2.1 Logistic regression

Since it’s a multi-class classification problem, a softmax function is used to generate probabilities; cross entropy loss is used as the loss function and gradient descent for optimization.  

Parameter setting: learning_rate = 0.01; training_stes = 500; batch_size = 128.

After running 500 steps, the accuracy on the training and testing dataset are 0.1718 and 0.1193 respectively.

### 2.2.2 Convolutional Neural Networks

Parameter setting: learning_rate = 0.01; trainning_steps = 500; batch_size = 128; dropout = 0.7  
conv1_filters = 16, conv2_filters = 32, conv3_filters = 128, fc1_units = 1024.

The architecture of the CNN model built in the task is: 
![CNN architecture](https://github.com/fangyiyu/Fangyi_Yu_Modified_MNIST/blob/master/CNN%20architecture.png)

Softmax_cross_entropy_loss is used as the loss function, and adam as the optimizer. After running 500 steps, the accuracy on the training and testing dataset are 0.1250 and 0.1116 respectively.  

### 2.2.3 Gated Convolutional Neural Networks

Parameter setting is the same as in the CNN model.
The convolutional layer in GCN used a gating mechanism to allow the network to control what information should be propagated the hierarchy of layers. So I built the gated convolutional layer based on the mechanism below:

![Gating mechanism](https://github.com/fangyiyu/Fangyi_Yu_Modified_MNIST/blob/master/Gating%20mechanism.png)

The architecture of the GCN model built in this task is the same as the CNN model above, and the loss function and optimizer are also identical. After running 500 steps, the accuracy on the training dataset is 0.0703, while when predicting using the testing dataset, it showed my GPU was out of memory even after decreasing units in the hidden layer or adding strides in the Maxpooling layers.  

### 2.2.4 Recurrent Neural Networks

Parameter setting: learning_rate = 0.01; epochs=30; batch_size = 128  
A simple one layer LSTM model is built for this task with categorical_crossentropy as the loss function and rmsprop as the optimizer.

After running 30 epochs, the accuracy on the training and testing dataset are 0.2582 and 0.2288respectively.

### 2.2.5 VGG

Apart from the vanilla CNN model, I also used VGG on the dataset and got a decent performance.

Data augmentation was first implemented to increase the diversity of the training set, and then the VGG model was built using keras. After training for 15 epochs, the accuracy on the training and testing dataset are 0.9283 and 0.9689 respectively.

# 3. Conclusion

A simple CNN, RNN or GCN model is not powerful enough to capture the patterns in the training dataset, in other words, they underfit the modified MNIST dataset, to address underfitting, we could try using a bigger neural network by adding new layers or increasing the number of neurons in existing layers or training the models for longer; therefore, I implemented a more sophiscated model-VGG on the dataset, and it turned out to fit the dataset well. 

*References:*  
[1] https://www.dropbox.com/sh/jn8p1pvpgjy3b9b/AABWc6ouePh2YJFZkGA9zE3ha?dl=0  
[2] https://github.com/aymericdamien/TensorFlow-Examples#tensorflow-examples  
[3] https://arxiv.org/pdf/1612.08083.pdf
