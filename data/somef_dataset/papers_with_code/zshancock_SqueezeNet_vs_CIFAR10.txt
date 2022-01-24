# SqueezeNet vs. CIFAR10
## Zac Hancock (zshancock@gmail.com)


## Introduction

SqueezeNet is focused on size and performance over outright accuracy, however, it still achieved AlexNet-level accuracy in the paper by Iandola in 2016. The actual SqueezeNet architecture is different than what I will refer to as 'Squeeze Net' so I encourage you to read the paper (cited below) and visit the [Deepscale/SqueezeNet github page](https://github.com/deepscale/squeezenet). My model did not reach [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)-level accuracy (89%) but did reach approximately 80% with only 122k parameters (AlexNet is ~ 60million, VGG is 130million+). Additionally, my model is much smaller than even that referenced in the SqueezeNet paper. 

## The CIFAR-10 Data

This is a baseline data set of tiny 32 x 32 x 3 images with 10 classes. Because of its size, it was appropriate for my local machine. There are 50,000 training images and 10,000 training images.  

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/cifar_visual.JPG)

**25 Images from the CIFAR-10 dataset (notice: this only shows 9 of the 10 classes - no 'bird' shown).**

## The Model

Inspired by the 'SqueezeNet' architecture proposed by Forrest Iandola et al. (2016), created a smaller model for CIFAR-10 data set using similar components (fire module, etc). The basis of the **fire module** is shown below (Iandola 2016). Essentially, the fire module implements a strategy wherein it minimizes the input parameters by utilizing a 'squeeze layer' that only uses 1x1 filters. After the 'squeeze layer' is a series of both 1x1 and 3x3 filters in the 'expand layer'. The expand layer is then concatenated. 

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/fire_module.JPG)

```
def fire_mod(x, fire_id, squeeze=16, expand=64):
    
    # initalize naming convention of components of the fire module
    squeeze1x1 = 'squeeze1x1'
    expand1x1 = 'expand1x1'
    expand3x3 = 'expand3x3'
    relu = 'relu.'
    fid = 'fire' + str(fire_id) + '/'
    
    # define the squeeze layer ~ (1,1) filter
    x = layers.Convolution2D(squeeze, (1,1), padding = 'valid', name= fid + squeeze1x1)(x)
    x = layers.Activation('relu', name= fid + relu + squeeze1x1)(x)
    
    # define the expand layer's (1,1) filters
    expand_1x1 = layers.Convolution2D(expand, (1,1), padding='valid', name= fid + expand1x1)(x)
    expand_1x1 = layers.Activation('relu', name= fid + relu + expand1x1)(expand_1x1)
    
    # define the expand layer's (3,3) filters
    expand_3x3 = layers.Convolution2D(expand, (3,3), padding='same', name= fid + expand3x3)(x)
    expand_3x3 = layers.Activation('relu', name= fid + relu + expand3x3)(expand_3x3)
    
    # Concatenate
    x = layers.concatenate([expand_1x1, expand_3x3], axis = 3, name = fid + 'concat')
    
    return x

```

Using the fire module outlined above, the architecture was completed. Max Pooling happens after the very first convolution layer, followed by 4 fire modules. After the last fire module, 50% dropout is committed before the last convolution layer. Global pooling is committed right before softmax activation into 10 classes. The original SqueezeNet proposed by Iandola was much larger, but the CIFAR images are considerably smaller than ImageNet ~ additionally, my local machine could struggle with a larger model. 

```
def SqueezeNet(input_shape = (32,32,3), classes = 10):
        
    img_input = layers.Input(shape=input_shape)
    
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model
```

## Results

The first training of this architecture after 10 epochs resulted in near 50% training accuracy, without overfitting. However, better accuracy could be achieved once increased to 250 epochs. Admittedly, it does look like some overfitting was beginning to occur, as the training accuracy and loss begins to get stronger than the testing metrics. The model was impressively lightweight and worked on a very average machine, even at 250 epochs. I might come back to try to construct the entire SqueezeNet as documented (more fire modules, bypass, etc).  

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/accuracy_and_loss.JPG)
**Results of the model versus the CIFAR-10 on only 10 epochs**

![alt text](https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/graphics/250_epochs_acc_and_loss.JPG)
**Results of the model versus the CIFAR-10 on 250 epochs**


## Citations

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

```
Author = Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally and Kurt Keutzer
Title = SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
Journal = {arXiv:1602.07360}
Year = 2016
```

CIFAR-10 Documentation ~
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

```
Author = Alex Krizhevsky
Title = Learning Multiple Layers of Features from Tiny Images
Year = 2009
```

AlexNet paper -
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

```
Author = Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
Title = ImageNet Classification with Deep Convolutional Neural Networks
Year = 2012
```
