# CompVision
### MNIST_1
<b>Layer-1</b>: Conv2d (5x5 kernels, 32 filters) + Max pooling (2, 2)  <br>
<b>Layer-2</b>: Conv2d (5x5,64 filters) + Max pooling (2, 2) + dropout<br>
<b>Layer-3</b>: Linear (1024 inputs, 10 outputs) + RelU <br>
<br>
### MNIST_2
<b>Layer-1</b>: Conv2d (5x5 kernels, 32 filters) + Max pooling (2, 2) <br>
<b>Layer-2</b>: Conv2d (5x5,64 filters) + Max pooling (2, 2) + dropout <br>
<b>Layer-3</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-4</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) + Output of Layer-2 (Residual) <br>
<b>Layer-5</b>: Linear (1024 inputs, 10 outputs) + RelU <br>
### MNIST_3
<b>Layer-1</b>: Conv2d (5x5 kernels, 32 filters) + Max pooling (2, 2) <br>
<b>Layer-2</b>: Conv2d (5x5,64 filters) + Max pooling (2, 2) + dropout <br>
<b>Layer-3</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-4</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-5</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-6</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-7</b>: Linear (1024 inputs, 10 outputs) + RelU <br>
### MNIST_4
<b>Augmentation</b>: Random rotations + normalization<br>
<b>Layer-1</b>: Conv2d (3x3 kernels, 32 filters) <br>
<b>Layer-2</b>: Conv2d (3x3,32 filters) + Max pooling (2, 2) + dropout <br>
<b>Layer-3</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) <br>
<b>Layer-4</b>: Conv2d (3x3 kernels, 64 filters, padding = 1) + Max pooling (2, 2) + dropout<br>
<b>Layer-5</b>: Conv2d (3x3 kernels, 128 filters, padding = 1) + Max pooling(2, 2) + BN<br>
<b>Layer-6</b>: Linear (1152 inputs, 10 outputs) + RelU <br>
<i>Heavy dropouts countered by adding more channels, training on more epochs and pooling later than previous models</i> 
### Neural Style Transfer
<b>Pytorch tutorial</b> - https://pytorch.org/tutorials/advanced/neural_style_tutorial.html<br>
A few examples in <b>Sherlock_Examples.ipynb</b> (With different weights for style loss and content loss)<br>
### MNIST_SpatialTransformer
<b>Paper</b>: https://arxiv.org/abs/1506.02025<br>
MNIST-4 with an intermediate spatial transformer layer<br>
### Flipkart - Object detection
Finding objects in a given image using object detection in PyTorch for the <a href="https://dare2compete.com/o/Flipkart-GRiD-Teach-The-Machines-2019-74928">contest</a>. 
