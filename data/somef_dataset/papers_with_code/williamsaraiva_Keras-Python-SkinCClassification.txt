# Keras-Python-SkinCClassification
Maleficent Melanoma Classification 



# Keras VGG16 - 
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
VGG16 model, with weights pre-trained on ImageNet.

This model can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).

The default input size for this model is 224x224.

Arguments
include_top: whether to include the 3 fully-connected layers at the top of the network.
weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be  (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 48. E.g. (200, 200, 3) would be one valid value.
pooling: Optional pooling mode for feature extraction when include_top is False.
None means that the output of the model will be the 4D tensor output of the last convolutional layer.
'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
'max' means that global max pooling will be applied.
classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
Returns
A Keras Model instance.

References
Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/abs/1409.1556

License
These weights are ported from the ones <a href='http://www.robots.ox.ac.uk/~vgg/research/very_deep/'>released by VGG at Oxford</a> under the <a href='https://creativecommons.org/licenses/by/4.0/'>Creative Commons Attribution License.</a>

Source font: https://keras.io/applications/#vgg16
