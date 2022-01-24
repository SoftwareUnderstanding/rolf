# FlowerClassification

# Introduction

---

Here I have taken a flower classification dataset from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. After some preprocessing of images I took InceptionNet as my model and fine tuned it's last 50 layers and freezed all the first 50 but batch normalization layers.

# About Inception Versions

---

The Inception network , was complex (heavily engineered). It used a lot of tricks to push performance; both in terms of speed and accuracy. Its constant evolution lead to the creation of several versions of the network. The popular versions are as follows:
 - Inception v1.
 - Inception v2 and Inception v3.
 - Inception v4 and Inception-ResNet.
Each version is an iterative improvement over the previous one. Understanding the upgrades can help us to build custom classifiers that are optimized both in speed and accuracy. Also, depending on your data, a lower version may actually work better.

## Inception V1 (Paper : https://arxiv.org/pdf/1409.4842v1.pdf)

---

### The Premise:

 - Salient parts in the image can have extremely large variation in size.
 - Because of this huge variation in the location of the information, choosing the right kernel size for the convolution operation becomes tough. A larger kernel is preferred for information that is distributed more globally, and a smaller kernel is preferred for information that is distributed more locally.
 - Very deep networks are prone to overfitting. It also hard to pass gradient updates through the entire network.
 - Naively stacking large convolution operations is computationally expensive.
 
### The Solution:

Why not have filters with multiple sizes operate on the same level? The network essentially would get a bit “wider” rather than “deeper”. The authors designed the inception module to reflect the same.

The naive inception model performs convolution on an input, with 3 different sizes of filters (1x1, 3x3, 5x5). Additionally, max pooling is also performed. The outputs are concatenated and sent to the next inception module. https://miro.medium.com/max/1400/1*DKjGRDd_lJeUfVlY50ojOA.png 

As stated before, deep neural networks are computationally expensive. To make it cheaper, the authors limit the number of input channels by adding an extra 1x1 convolution before the 3x3 and 5x5 convolutions. Though adding an extra operation may seem counterintuitive, 1x1 convolutions are far more cheaper than 5x5 convolutions, and the reduced number of input channels also help. Do note that however, the 1x1 convolution is introduced after the max pooling layer, rather than before. 
https://miro.medium.com/max/1400/1*U_McJnp7Fnif-lw9iIC5Bw.png

Using the dimension reduced inception module, a neural network architecture was built. This was popularly known as GoogLeNet (Inception v1).

GoogLeNet has 9 such inception modules stacked linearly. It is 22 layers deep (27, including the pooling layers). It uses global average pooling at the end of the last inception module.

Needless to say, it is a pretty deep classifier. As with any very deep network, it is subject to the vanishing gradient problem.
To prevent the middle part of the network from “dying out”, the authors introduced two auxiliary classifiers (The purple boxes in the image). They essentially applied softmax to the outputs of two of the inception modules, and computed an auxiliary loss over the same labels. The total loss function is a weighted sum of the auxiliary loss and the real loss. Weight value used in the paper was 0.3 for each auxiliary loss.

`# The total loss used by the inception net during training.`

`total_loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2`

## Inception V2 (https://arxiv.org/pdf/1512.00567v3.pdf)

Inception v2 and Inception v3 were presented in the same paper. The authors proposed a number of upgrades which increased the accuracy and reduced the computational complexity.

### The Premise:

 - Reduce representational bottleneck. The intuition was that, neural networks perform better when convolutions didn’t alter the dimensions of the input drastically. Reducing the dimensions too much may cause loss of information, known as a “representational bottleneck”
 - Using smart factorization methods, convolutions can be made more efficient in terms of computational complexity.
 
### The Solution:

 - Factorize 5x5 convolution to two 3x3 convolution operations to improve computational speed. Although this may seem counterintuitive, a 5x5 convolution is 2.78 times more expensive than a 3x3 convolution. So stacking two 3x3 convolutions infact leads to a boost in performance. https://miro.medium.com/max/1114/1*RzvmmEQH_87qKWYBFIG_DA.png
 - Moreover, they factorize convolutions of filter size nxn to a combination of 1xn and nx1 convolutions. For example, a 3x3 convolution is equivalent to first performing a 1x3 convolution, and then performing a 3x1 convolution on its output. They found this method to be 33% more cheaper than the single 3x3 convolution. https://miro.medium.com/max/1196/1*hTwo-hy9BUZ1bYkzisL1KA.png
 - The filter banks in the module were expanded (made wider instead of deeper) to remove the representational bottleneck. If the module was made deeper instead, there would be excessive reduction in dimensions, and hence loss of information. https://miro.medium.com/max/1150/1*DVXTxBwe_KUvpEs3ZXXFbg.png
 - The above three principles were used to build three different types of inception modules (Let’s call them modules A,B and C in the order they were introduced. 

## Inception V3

### The Premise:

 - The authors noted that the auxiliary classifiers didn’t contribute much until near the end of the training process, when accuracies were nearing saturation. They argued that they function as regularizes, especially if they have BatchNorm or Dropout operations.
 - Possibilities to improve on the Inception v2 without drastically changing the modules were to be investigated.

### The Solution

Inception Net v3 incorporated all of the above upgrades stated for Inception v2, and in addition used the following:
 - RMSProp Optimizer.
 - Factorized 7x7 convolutions.
 - BatchNorm in the Auxillary Classifiers.
 - Label Smoothing (A type of regularizing component added to the loss formula that prevents the network from becoming too confident about a class. Prevents over fitting).
 
## Inception V4 (Inception v4 and Inception-ResNet were introduced in the same paper. https://arxiv.org/pdf/1602.07261.pdf)

### The Premise:

 - Make the modules more uniform. The authors also noticed that some of the modules were more complicated than necessary. This can enable us to boost performance by adding more of these uniform modules.
The Solution
 - The “stem” of Inception v4 was modified. The stem here, refers to the initial set of operations performed before introducing the Inception blocks. https://miro.medium.com/max/1292/1*cYjhQ05zLXdHn363TsPrLQ.jpeg
 - They had three main inception modules, named A,B and C (Unlike Inception v2, these modules are infact named A,B and C). They look very similar to their Inception v2 (or v3) counterparts. https://miro.medium.com/max/1400/1*KrBAIZjcrlXu6JPiPQj2vQ.jpeg
 - Inception v4 introduced specialized “Reduction Blocks” which are used to change the width and height of the grid. The earlier versions didn’t explicitly have reduction blocks, but the functionality was implemented. https://miro.medium.com/max/1400/1*2Hdo2wG3ILUoYaorJdlR-Q.jpeg

**Note: I have used inception v3 as my model.**

## Model Accuracy
 - `Train Accuracy: 100%`
 - `Test Accuracy: 95.99%`




