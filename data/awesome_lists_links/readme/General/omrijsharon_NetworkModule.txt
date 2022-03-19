# NetworkModule
An easy way to create a fully connected network with pytorch. 
This module contains an additional function modules that can be used with pytorch Sequential.

## How to use the module?

The module gets a list of layers and a list of activation functions.
In the layer's list, each element corresponds to the number of nodes in the layer, and the length of the list is the number of layers in the network.

i.e.: the code
```
L = [16, * 2 * [8] , 4]
activation_func = [*(len(L)-2) * [functional.SeLU()], functional.Identity()]
```
will help us to create a network with:
1. input layer: 16 nodes
2. hidden layer: 8 nodes, with SeLU as activation function 
3. hidden layer: 8 nodes, with SeLU as activation function 
4. output layer: 4 nodes, with a Identity activation function

notice that Identity() is a linear activation function. It is exacly like not putting any activation function on the layer. Yet, it is necessary that each layer will have an activation function, except for the 1st/Input layer. In other words, *when no activation function is needed, use functional.Identity()*.

Also notice that the length of the activation_func list is always smaller by 1 than the layers' list length , because the 1st layer never gets an activation function.



**example:**

```
from NetworkModule import Network
from NetworkModule import functional as functional
import torch.nn as nn

input_dim = 16
output_dim = 4
hidden_layers = 2*[8]
L = [input_dim, *hidden_layers, output_dim]
activation_func = [functional.SeLU(), functional.Sin(), nn.Softmax(dim=1)]
net = Network(L, activation_func, dropout=0.5)
```

The network can also use dropout. In this example, the dropout probability is set to 0.5.
- IMPORTANT: dropout should not be used on the weights between the last 2 layers. In the last example we have 4 layers. Dropout will be activated only on the weights between layers 1-2 and 2-3.

## functional module
This module is an extension to torch.nn module. 
You can use these functions like you would use nn.Sigmoid(). Most convenient in torch Sequential:
### SeLU
```
functional.SeLU()
```
Paper: https://arxiv.org/pdf/1706.02515.pdf , TL;DR: like leaky ReLU, without the problems of exploding/vanishing gradients.
### Sine
```
functional.Sin()
```
a sin function that can be used in Sequential.
### Cosine
```
functional.Cos()
```
a cos function that can be used in Sequential.
### Identity/Linear
```
functional.Identity()
```
a linear activation function that does nothing to its input and can be used in Sequential.
