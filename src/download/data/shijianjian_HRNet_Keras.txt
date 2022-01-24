# HRNet_Keras

The implementation of [HRNet](https://github.com/HRNet/HRNet-Image-Classification) with [SE modules](https://github.com/hujie-frank/SENet) for image classification tasks.

## Usage
As illastrated in the main function. You may use it as below.
``` python
model = SE_HRNet(blocks=3, reduction_ratio=4, init_filters=32, training=True).build(input_shape=(2048, 2048, 3), num_output=2, repetitions=4)
```
