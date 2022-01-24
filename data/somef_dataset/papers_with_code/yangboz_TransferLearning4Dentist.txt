```
I think AI is akin to building a rocket ship. 

You need a huge engine and a lot of fuel. If you have a large engine and a tiny amount of fuel, you won’t make it to orbit.

If you have a tiny engine and a ton of fuel, you can’t even lift off.

To build a rocket you need a huge engine and a lot of fuel.

The analogy to deep learning is that the rocket engine is the deep learning models and the fuel 
is the huge amounts of data we can feed to these algorithms. — Andrew Ng
```

# TransferLearning4Dentist

Transfer Learning for Dentist diagnosis aid , an intelligent healthcare application for train/test/predict.

[](https://cdn-images-1.medium.com/max/800/0*ovwBU8FJHCqqvsOr.gif)

## LeNet

![LeNet](https://raw.githubusercontent.com/yangboz/TransferLearning4Dentist/master/LeNet/plot.png)

## AlexNet

## ZfNet

## VggNet

## ResNet

![ResNet](https://raw.githubusercontent.com/yangboz/TransferLearning4Dentist/master/ResNet/plot.png)

### Demo

http://118.190.96.120/iDentisty/client/

## Overview

(VGG16) model Transfer Learning, + Keras ImageDataGenerator, fine-tune on VGG16, base on MobileNet with ImageNet weights, for prediction 

and evaluate the final score.

## Conceptual Framework & Details

### VGG16

![vgg16](https://blog.keras.io/img/imgclf/vgg16_original.png)

### VGG16+

![vgg16+](https://blog.keras.io/img/imgclf/vgg16_modified.png)


## References

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab

http://cs231n.github.io/transfer-learning/

https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

https://medium.com/datadriveninvestor/keras-imagedatagenerator-methods-an-easy-guide-550ecd3c0a92

## Papers

1.Lenet，1986：http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf 

2.Alexnet，2012：http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf 

3.VGG，2014：https://arxiv.org/pdf/1409.1556.pdf 

4.GoogleNet，2014：https://arxiv.org/pdf/1409.4842.pdf 

5.ResNet，2015：https://arxiv.org/pdf/1512.03385.pdf

