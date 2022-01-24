# Neural networks in python

This code is intended mainly as proof of concept of the algorithms presented in [1-7]. The implementations are not particularly clear, efficient, well tested or numerically stable. We advise against using this software for nondidactic purposes.

This software is licensed under the MIT License. 

## Models

* Feedforward neural network (classifier)
    * Arbitrary number of layers
    * Cross-entropy and negative log-likelihood cost functions
    * Backpropagation
    * Stochastic gradient descent
    * L2 regularization
    * Momentum

* Recurrent neural network (sequence element classifier)
    * Single hidden layer
    * Cross-entropy and negative log-likelihood cost functions
    * Backpropagation
    * Online gradient descent
    * L2 regularization
    * Momentum

* Long Short-Term memory network (sequence element classifier)
    * Single hidden layer
    * Cross-entropy and negative log-likelihood cost functions
    * Backpropagation
    * Online gradient descent
    * Momentum

* Restricted Boltzmann machine (probabilistic graphical model)
    * Gibbs sampling
    * k-step persistent contrastive divergence
    * Stochastic gradient descent
    * Momentum

## Examples

See the examples directory. 

Some examples use keras, a neural networks library.

## References

[1] Nielsen, Michael. Neural Networks and Deep Learning, Available in http://neuralnetworksanddeeplearning.com, 2015.

[2] Hinton, Geoffrey. Neural Networks for Machine Learning, Available in http://www.coursera.org, 2012.

[3] Li, Fei-Fei and Karpathy, Andrej. Convolutional Neural Networks for Visual Recognition, Available in http://cs231n.github.io/convolutional-networks, 2015.

[4] Simonyan, Kare, and Zisserman, Andrew. Very deep convolutional networks for large-scale image recognition, Available in http://arxiv.org/abs/1409.1556, 2014.

[5] Graves, Alex. Supervised Sequence Labelling with Recurrent Neural Networks, 2012.

[6] Fischer, Asja and Igel, Christian. An Introduction to Restricted Boltzmann machines. CIARP, 2012.

[7] Hinton, Geoffrey. A Practical Guide to Training Restricted Boltzmann Machines, 2010.
