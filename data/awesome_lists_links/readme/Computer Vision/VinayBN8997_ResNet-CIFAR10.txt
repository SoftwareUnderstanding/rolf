# ResNet-CIFAR10
A notebook referenct to model ResNet on CIFAR 10

### In this post,  we are tarining a ResNet network on CIFAR10. The ResNet model used is pretrained on the ImageNet dataset.

To import the pretarined model, we are using another GitHub repository from Pavel Yakubovskiy.
Link: https://github.com/qubvel

The model is trained on Google Colab which provides 12 hours of free GPU instance per session.

To clone the model into python library:
```python
!pip install git+https://github.com/qubvel/classification_models.git
```
Requirements:
1. Keras
2. Classification_models : Github: https://github.com/qubvel/classification_models.git
3. Numpy
4. Matplotlib
5. OpenCV



## About training dataset:

The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. CIFAR-10 is a labeled subset of the 80 million tiny images dataset.

Source: https://en.wikipedia.org/wiki/CIFAR-10


## Network architecture: ResNet

arXiv: https://arxiv.org/abs/1512.03385

It is observed that as the networks goes deeper and deeper, during the convergence, the degradation of weights is an inevitable problem. The weights get too small which leads to saturated accuracy.
To avoid this problem, skip connections are introduced into the architecture so that instead of just stacking up of layers, the prior reidual mapping is also concatenated with the current mapping so that the architecture is explicitly let to fit a residual mapping.
Below is a Residual block used in the ResNet architecture. Here the identity mapping of input X is also added to the output of the convolution block. On doing this in all the convolution blocks, the degradation problem is tackled.

![image](https://user-images.githubusercontent.com/33830482/61840489-f89b9880-aeae-11e9-809d-8eaa7befdf9a.png)

To get the model configured from ImageNet to CIFAR10 configuration, we need to add anothe layer at the end.

```python
base_model = ResNet18(input_shape=(32,32,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
```

## Next, we are working on Gradcam which helps in understanding what the model is looking at

Reference: http://www.hackevolve.com/where-cnn-is-looking-grad-cam/

Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept (say logits for ‘dog’ or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
Process:

1. Compute the gradient of the class output value with respect to the feature map
2. Pool the gradients over all the axes leaving out the channel dimension
3. Weigh the output feature map with the computed gradient values
4. Average the weighed feature map along the channel dimension resulting in a heat map of size same as the input image
5. Finally normalize the heat map to make the values in between 0 and 1

### 3 funtions are written which returns the activation map from thier respective layers as below:

1. stage1_unit1_relu2 : Initial stage of the network
2. stage1_unit2_relu2 : Layer approximately in the middle of the architecture
3. stage4_unit1_relu1: Deeper stage of the network






