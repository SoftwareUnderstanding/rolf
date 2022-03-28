# tf_conv_cosnorm
Tensorflow implementation of cosine normalization [1] for convolutional layer.
There is no implementation detail of convolution version in the original paper [1].
Therefore, the performance of this implementation is not guaranteed.

## Usage
```
# Example of using tf.nn.conv2d
conv = tf.nn.conv2d(x, w, strides, padding)
relu = tf.nn.relu(conv + bias)

# conv2d_cosnorm
conv = conv2d_cosnorm(x, w, strides, padding)
relu = tf.nn.relu(conv) # No bias needed
```

```
# Example of using Keras
from keras.layers import Conv2d
model.add(Conv2D(64, (3, 3)))

# conv2d_cosnorm
from sim_layer import Norm_Conv2d as Conv2D
model.add(Conv2D(64, (3, 3)))
```

## Test
A modified version of test code [2] is used to dubug and test of this implementation.

## References
- [1] Chunjie, Luo, and Yang Qiang. "Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks." (https://arxiv.org/abs/1702.05870)
- [2] https://github.com/aymericdamien/TensorFlow-Examples