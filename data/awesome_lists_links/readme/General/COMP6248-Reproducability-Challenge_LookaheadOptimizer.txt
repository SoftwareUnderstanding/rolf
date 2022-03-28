# LookaheadOptimizer
Reproduction of CIFAR-10/CIFAR-100 and Penn Treebank experiments to test claims in [LookaheadOptimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610) 


## Introduction 
Stochastic gradient descent (SGD) is a popular method for training neural networks, using “minibatches” of data to update the network’s weights.  Improvements are often made upon SGD by either using acceleration/momentum or by altering the learning rate over time. Zhang et al. propose a novel improvement in [LookaheadOptimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610). They showed that Lookahead consistantly outperformed other optimizers on popular language modelling, machine translantion and image classification tasks. 

Lookahead uses a set of fast weights which lookahead k steps and a set of slow weights with learning rate alpha. From a high-level perspective, Lookahead chooses the search direction by calculating the fast weights of the inner optimizer. The approach facilitates the use of any inner optimizer such as Adam or SGD. This comes with the cost of a slighlty increased time complexity, however the original authors illustrate a significant increase in efficiency. 

This project aims to test these findings by reimplementing the main CIFAR-10/100 and Penn Treebank experiments. See [our paper](https://github.com/COMP6248-Reproducability-Challenge/LookaheadOptimizer/blob/master/Reproducibility%20Challenge%20LA%20Optimizer.pdf) for more details and our findings.


## Team Members 

Connah Romano-Scott - crs1u19@soton.ac.uk

John Joyce - jvjj1u19@soton.ac.uk

Ilias Kazantzidis - ik3n19@soton.ac.uk

## Code References 

All experiments used the PyTorch implementation of Lookahead (written by Zhang et al.), available at: https://github.com/michaelrzhang/lookahead

The ResNet-18 implementation in this work (and Zhang et al.'s work) is availble at: https://github.com/uoguelph-mlrg/Cutout/blob/master/model/resnet.py

The Penn Treebank training setup in this work (and Zhang et al.'s work) is modified from: https://github.com/salesforce/awd-lstm-lm
See [Penn ReadMe](https://github.com/COMP6248-Reproducability-Challenge/LookaheadOptimizer/blob/master/PTB/README.MD) for more information regarding the modifications of this code. 
