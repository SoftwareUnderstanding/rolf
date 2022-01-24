# DenseNet-Keras
Implementation of DenseNet in Keras with Tensorflow backend

This is an ipython Notebook demonstrating the results of current DenseNet implementation. In Future, this will be extended to be imported as a python package. Training time is 10+ hrs for 250 Epoch on 1080Ti GPU.

## Source:    
Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993

![Alt text](images/dense_net_graphic.JPG?raw=true "Densenet image from paper")

## Abstract:
*Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance.*

## Objectives:
1. To study the DenseNet Architecture and the Bottleneck-Compressed modification in it.
2. To deepen understanding of data augmentation and its regularizing impact.
3. To use Stochastic Gradient Descent as optimizer and experiment with various learning_rate/momentum strategies.

## Results:
We achieve 91% accuracy as compared to Paper's 93% (without augmentation) and 94.7% (with augmentation) for CIFAR10 database.

## Future Work:
- Implement DenseNet with k=12, Depth=40, Param=1M & achieve 93% accuracy with SGD in 300 epochs.
- Implement DenseNet-BC with k=12, Depth=100, Param=0.8M & study any increase in accuracy.

## SGD:
lr=0.1, decay=1e-4, momentum=0.9, nesterov=True

### Epoch 1 to 150: lr=0.1

![Alt text](images/accuracy_epoch_1_150.png?raw=true "Accuracy for Epoch 1 to 150")

### Epoch 151 to 200: lr=0.01

![Alt text](images/accuracy_epoch_151_200.png?raw=true "Accuracy for Epoch 151 to 200")

### Epoch 201 to 250: lr=0.0001

![Alt text](images/accuracy_epoch_201_250.png?raw=true "Accuracy for Epoch 201 to 250")

## Requirements:
- Keras
- Tensorflow
- numpy
- matplotlib
