# CIFAR-10-object-detection
Object detection of CIFAR-10 using densenet

# Rendered Notebook: https://nbviewer.jupyter.org/github/VinayBN8997/CIFAR-10-object-detection/blob/master/CIFAR%2010%20object%20detection.ipynb

CNN Architecture: Densely Connected Convolutional Networks
Source: https://arxiv.org/abs/1608.06993

## Major insights from the paper:
1. To further improve the information flow between layers inside a dense block, a different connectivity pattern is introduced which are direct connections from any layer to all subsequent layers.
2. Each dense block has a composite function of three consecutive operations as batch normalization (BN), followed by a rectified linear unit (ReLU) and a 3 × 3 convolution (Conv).
3. The transition layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer.
4. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, e.g., k = 12. Here k is the growth rate which represents the number of feature maps for each layer of convolution.
5. A 1×1 convolution is be introduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency.
6. To further improve model compactness, the number of feature-maps at transition layers is reduced uisng a compression factor. If a dense block contains m feature-maps, the following transition layer generate θ*m output featuremaps.
7. At the end of the last dense block, i.e, in the output block, a global average pooling is performed and then a softmax classifier is attached.

## About Data:
The CIFAR-10 *dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains *60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. CIFAR-10 is a labeled subset of the 80 million tiny images dataset.

Source: https://en.wikipedia.org/wiki/CIFAR-10

# Steps:
1. Importing data and converting output variable to one-hot vector form
2. Normalising Data
3. Data Augmentation
4. Defining blocks: Dense Block,Transition Block and Output Layer
5. Model implementation
6. Testing different learning rate decay algorithms
7. Remarks
