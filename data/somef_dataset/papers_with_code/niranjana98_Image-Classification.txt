# Image Classification

## Datasets
### CALTECH 101
This Dataset contains pictures of objects belonging to 101 categories. About 40 to 800 images are present per category. Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato.  The size of each image is roughly 300 x 200 pixels.

### COIL100
COIL-100 was collected by the Center for Research on Intelligent Systems at the Department of Computer Science, Columbia University. The database contains color images of 100 objects. The objects were placed on a motorized turntable against a black background and images were taken at pose internals of 5 degrees. This dataset was used in a real-time 100 object recognition system whereby a system sensor could identify the object and display its angular pose. There are 7,200 images of 100 objects. Each object was turned on a turnable through 360 degrees to vary object pose with respect to a fixed color camera. Images of the objects were taken at pose intervals of 5 degrees. This corresponds to 72 poses per object. There images were then size normalized. Objects have a wide variety of complex geometric and reflectance characteristics.

### Wang
This dataset contains 1000 images in 10 categories. Every 100 image belongs to a category.

## Architectures
### VGGNet
VGG stands for Visual Geometric Group. It takes a 224x224x3 image as input. VGG Net consists of architectures ranging from VGG11 to Vgg22. The Convolutions layers use a 3x3 filter and also use 1x1 filters for linear transformation with the stride of 1. It uses ReLu transformation function.

#### VGG16
VGG 16 consists of 16 weighted layers consisting of 13 Convolutional Layers and 3 Dense Layers (Configuration D)

#### VGG19
VGG 19 consists of 19 weighted layers consisting of 16 Convolutional Layers and 3 Dense Layers (Configuration E)

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/VGGNet.png)
Figure: Different VGGNet Architectures

### Inception
The main idea behind the Inception Architecture is to effectively detect and classify images which have large variations insize of the region of interest. Since deep networks have a problem of overfitting and are computationally complex, it is more efficient to make a wider network with different convolutional nets with different filter sizes such as 3x3, 5x5 at the same level. And the outputs from each level is concatenated. There are 4 versions of Inception. 

#### Inception V3
Inception V3 consists of inception blocks which contains 4 braches which process the inputs: a 1x1 CONV layer, a pooling layer followed by a 1x1 layer, a 1x1 layer and a 3x3 layer and finally a 1x1 layer followed by 2 3x3 layers (which is equivalent to 5x5 layer). Since such architectures have the problem of "Vanishing Gradients", the technique of auxillary loss is used, in which output is obtained from 2 inception modules and loss is computed. The total loss function is a weighted sum of the auxiliary loss and the real loss. In such Auxillary classifiers, Batch Normalisation is used. Inception V3 uses a RMSPropOptimiser.

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/Inception%20Module.png)
Figure: Representation of a Inception Module

The Architecture consists of 2 sets of 3x3 Convolution Layers seperated by a pooling layer. This is followed by 3 Inception Modules ending with a Pooling Layer, Linear Layer and a Softmax Layer.

#### Inception ResNet V2
Incpetion ResNet V2 is a combination on the Inception Architecture along with the techniques using in Residual Nets. This architecture is an updation over Inception V4 in which it has a different stem stucture, Different structure for the Inception blocks, different Reduction Blocks and Hyperparameter changes. The Reduction Blocks are used to change the Height and Width of the grid which is introduced in Inception V4. In this Architecture, similar to ResNets, the output from the inception blocks are summed with its input before it is passed on the next layer. 

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/Inception%20Block%20-%20ResNet.png)

Figure: An Inception Block from ResNet V2 Architecture

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/ResNet%20Architecture.png)
Figure: Complete ResNet Architecture. V1 and V2 only changes with changes in the Inception Modules.

### DenseNet201
In the ResNet Architecture, the output of a layer is summed with the input by elementwise addition. In DenseNet, the input to each layer obtains additional input from all preceding layers. This is done using Concatenation. This leads to the network being thinner and compact. Each block of the DenseNet consists of a Batch Normalisation, ReLu and a 3x3 Convolution layer. The Bottleneck part consists of a Bathc Normalisation, ReLu and a 1x1 Convolution Layer. The Transition layer is made up of 1x1 Convolution Layer followed by Average Pooling. Each Dense Block is followed by a Transition Layer. The end of the last Dense Block consists of a Global Pooling layer followed by a Softmax Classifier. 

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/DenseNet.png)
Figure: A 5 layer DenseNet Architecture

DenseNet 201 consists of 32 Blocks.

### Xception
In normal convolution, depthwise convolution is followed by pointwise convolution. Depthwise convolution refers to channel wise spatial convolution. Piecewise convolution refers to 1x1 convolution for dimension change. In Xception, pointwise convolution is followed by depthwise convolution. It also consists of residual connections where the input to a layer is summed with the output. 

![alt text](https://github.com/niranjana98/Image-Classification/blob/main/Xception%20Architecture.png)
Figure: Xception Architecture

## References
1. CALTECH Dataset http://www.vision.caltech.edu/Image_Datasets/Caltech101/
2. COIL 100 Dataset http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php
3. Wang Dataset http://wang.ist.psu.edu/docs/related/
4. VGGNet https://arxiv.org/abs/1409.1556
5. Inception V3 https://arxiv.org/pdf/1512.00567v3.pdf
6. Inception ResNet V2 https://arxiv.org/pdf/1602.07261.pdf
7. DenseNet201 https://arxiv.org/abs/1608.06993
8. Xception https://ieeexplore.ieee.org/document/8099678

