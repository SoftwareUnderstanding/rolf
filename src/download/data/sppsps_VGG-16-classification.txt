# TENSORFLOW IMPLEMENTATION OF VGG CLASSIFICATION

## usage
```
python training.py
```
Running this script will both train the model, and show the metric and loss plot.

## Help Log
```
Usage: ipykernel_launcher.py [-h] [--learning_rate LEARNING_RATE]
                                [--dropout DROPOUT] [--regularizer REGULARIZER]
                                [--epoch EPOCH] [--batch_size BATCH_SIZE]
                                [--momentum MOMENTUM] [--patience PATIENCE]
optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        learning rate for SGD
  --epoch EPOCH     max # of epoch
  --batch_size BATCH_SIZE
                        # of batch size
  --momentum MOMENTUM     momentum for SGD
  --dropout DROPOUT
                        Dropout percentage for the layers
  --regularization REGULARIZATION
                        Regularization value for L2 
```                        
## Contributers:
- [Pranjal Sharma](https://github.com/sppsps)
- [Dhrubajit Basumatary](https://github.com/dhruvz9)

## REFERENCE
 - Title : Very Deep Convolutional Networks for Large-Scale Image Recognition <br />
 - Link : https://arxiv.org/abs/1409.1556 <br />
 - Author : Karen Simonyan, Andrew Zisserman <br />
 - Published : 10 Apr 2015 <br />

# Summary

## Introduction
This model is implementing the architecture and training techniques used in VGG paper. We used CIFAR10 dataset by keras.<br />The preprocessing :<br />
A) Dividing all the pixels of images by 225<br />
B) Converting y_train(labels) into float type.<br />
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.<br />

## Architecture 
![alt text](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)<br />
This model is made for 224x224x3 images, but in this model, it is implemented on 32x32x3 images, so the learable parameters changed from 134 million to 43 million.
The following are the output values of all the layers used:<br/>
Model: "sequential"

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 512)         1180160   
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359808   
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 4, 4, 512)         2359808   
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 4, 4, 512)         2359808   
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
batch_normalization (BatchNo (None, 512)               2048      
_________________________________________________________________
dense (Dense)                (None, 4096)              2101248   
_________________________________________________________________
dropout (Dropout)            (None, 4096)              0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 4096)              16384     
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 4096)              16384     
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              4097000   
_________________________________________________________________
batch_normalization_3 (Batch (None, 1000)              4000      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                10010     
_________________________________________________________________
activation (Activation)      (None, 10)                0         
_________________________________________________________________
```

So in the end, we get a softmax layer with 10 units of output which will be used to train with the labels.

## Loss function
Categorical crossentropy: It is a loss function that is used in multi-class classification tasks. These are tasks where an example can only belong to one out of many possible categories, and the model must decide which one.
Formally, it is designed to quantify the difference between two probability distributions.

<p align="center">
  <img src="assets/categoricalloss.png" width="500">
  </p>
  
Loss Graph for the dataset

<p align="center">
  <img src="assets/lossplot.png" width="600">
  </p>

## Training
For training this model, mini-batch gradient descent optimizer is used, with learning rate 0.01 and momentum 0.9. Categorical loss Function is used as the loss function, batch size of 64, and I ran it for 50 epochs, resulting in training accuracy of 83%, validation accuracy of 73%. An early stop can also be used with regard to validation loss.
Finally, We have plotted the accuracy vs epochs with loss function which is shown using matplotlib.pyplot in the loss_plot.py script.
