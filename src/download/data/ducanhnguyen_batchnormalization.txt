# batchnormalization
BatchNormalization tries to mitigate the problem of internal covariate shift, where parameter initialization and changes in the distribution of the inputs of each layer affects the learning rate of the network

Paper: https://arxiv.org/pdf/1502.03167v3.pdf

In hidden units, before the calculation of activation function, datapoints of mini-batches are normalized into zero-mean using batch normalization.

<img src="https://github.com/ducanhnguyen/batchnormalization/blob/master/img/batchnormalization.png" width="550">

# Environment

Mac osx, jre 1.8, pycharm 2018

# Experiments

Dataset: digit-recognizer

Note: As seen in the experiment, from epoch 50, overfitting occurs.

<img src="https://github.com/ducanhnguyen/batchnormalization/blob/master/img/accuracy.png" width="550">

<img src="https://github.com/ducanhnguyen/batchnormalization/blob/master/img/error.png" width="550">
