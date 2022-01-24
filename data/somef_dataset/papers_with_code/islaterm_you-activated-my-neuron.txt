# You activated my neuron!

![http://creativecommons.org/licenses/by/4.0/](https://i.creativecommons.org/l/by/4.0/88x31.png)

This work is licensed under a 
[Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

This repository contains the basic implementation of neurons and neural networks using ``PyTorch`` 
tensors.

## Activation functions

The activation functions are defined inside the ``neuron.activation_functions`` package.

### Rectifier Linear Unit (ReLU)

The relu function of a tensor T is the element-wise max between 0 and the appropriate element of T.
It's definition is given by the function ``relu``.

### Swish

The Swish function, as proposed by by Ramachandran et al. on their paper "Searching for Activation 
Functions" ([arXiv:1710.05941v2](https://arxiv.org/abs/1710.05941)).
The implementation is given by the ``swish`` function.
