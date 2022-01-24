# EIP Session-5 Assignment
Assignment:

    1. Find 50 misclassified images from your Session 4 submission model
    2. Run GradCam (http://www.hackevolve.com/where-cnn-is-looking-grad-cam/) on these images
    3. Create a gallery of your GradCam results
    4. Upload your Colab file to a public github repo, and
    5. Upload your GitHub Link here: https://tinyurl.com/yxt6x2qq (https://tinyurl.com/yxt6x2qq)
    6. You need to attempt this quiz before the next session starts: https://tinyurl.com/y2t2ux8z (https://tinyurl.com/y2t2ux8z)

## Since session 4 submission was not a proper standard or in some cases where few people have deleted it, ResNet was set as the standard architecture with CIFAR 10 as the stardard dataset.

# About dataset:

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. CIFAR-10 is a labeled subset of the 80 million tiny images dataset.

Source: https://en.wikipedia.org/wiki/CIFAR-10

# Requirements:

    Keras
    Classification_models : Github: https://github.com/qubvel/classification_models.git
    Numpy
    Matplotlib
    OpenCV
    
# Network architecture: ResNet

### arXiv: https://arxiv.org/abs/1512.03385
  
  It is observed that as the networks goes deeper and deeper, during the convergence, the degradation of weights is an inevitable problem. The weights get too small which leads to saturated accuracy.
  To avoid this problem, skip connections are introduced into the architecture so that instead of just stacking up of layers, the prior reidual mapping is also concatenated with the current mapping so that the architecture is explicitly let to fit a residual mapping.
  Below is a Residual block used in the ResNet architecture. Here the identity mapping of input X is also added to the output of the convolution block. On doing this in all the convolution blocks, the degradation problem is tackled.
  
![image](https://user-images.githubusercontent.com/52725044/60933099-83409d00-a2de-11e9-8fe6-7957f2425ff9.png)

It is trained on Imagenet and the input shape is configured to 32 x 32 x 3, which is the size of our CIFAR10 dataset

# How Gradcam works:

### Reference: Where CNN is looking? – Grad CAM (http://www.hackevolve.com/where-cnn-is-looking-grad-cam/)

  Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept (say logits for ‘dog’ or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
Process:

    1. Compute the gradient of the class output value with respect to the feature map
    2. Pool the gradients over all the axes leaving out the channel dimension
    3. Weigh the output feature map with the computed gradient values
    4. Average the weighed feature map along the channel dimension resulting in a heat map of size same as the input image
    5. Finally normalize the heat map to make the values in between 0 and 1

3 funtions are written which returns the activation map from thier respective layers as below:

    stage1_unit1_relu2 : Initial stage of the network
    stage1_unit2_relu2 : Layer approximately in the middle of the architecture
    stage4_unit1_relu1: Deeper stage of the network

