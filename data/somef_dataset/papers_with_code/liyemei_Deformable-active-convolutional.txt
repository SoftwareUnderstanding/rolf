# Compiler Environment
vs2013 windows10 64-bit  cuda8.0 cudDNN5

# Thanks to the code:
- https://github.com/unsky/Deformable-ConvNets-caffe
- https://github.com/jyh2986/Active-Convolution
- https://github.com/Longqi-S/Focal-Loss
# layers
It includes the deformable conv layer\focal loss\active-conv.
Others new layers ,pelease reference 
- https://github.com/liyemei/caffe-segnet
- https://github.com/liyemei/CRF-as-RNN
# Usage
## About the deformable convolutional layer

### The params in Deformable Convolution:


```
bottom[0](data): (batch_size, channel, height, width)
bottom[1] (offset): (batch_size, deformable_group * kernel[0] * kernel[1]*2, height, width)
```


### Define:


```
f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
the output of the DeformableConvolution layer:

out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
```

### Offset layer:


```
layer {
  name: "offset"
  type: "Convolution"
  bottom: "pool1"
  top: "offset"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 72
    kernel_size: 3
    stride: 1
    dilation: 2
    pad: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

### Deformable Convolutional layer:


```
layer {
  name: "dec"
  type: "DeformableConvolution"
  bottom: "conv1"
  bottom: "offset"
  top: "dec"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  deformable_convolution_param {
    num_output: 512
    kernel_size: 3
    stride: 1
    pad: 2
    engine: 1
    dilation: 2
    deformable_group: 4
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
## Active Convolution
ACU has 4 parameters(weight, bias, x-positions, y-positions of synapse). Even though you don't use bias term, the order will not be changed.

Please refer deploy file in models/ACU

If you want define arbitary shape of convolution,

1. use non SQUARE type in aconv_param
2. define number of synapse using kernel_h, kernel_w parameter in convolution_param
In example, if you want define cross-shaped convolution with 4 synapses, you can use like belows.


```
...
aconv_param{   type: CIRCLE }
convolution_param {    num_output: 48    kernel_h: 1    kernel_w: 4    stride: 1 }
...
```
When you use user-defined shape of convolution, you'd better edit aconv_fast_layer.cpp directly to define initial position of synapses.

##  Focal Loss layer

```
optional SoftmaxFocalLossParameter softmax_focal_loss_param = XXX; (XXX is determined by your own caffe)

message SoftmaxFocalLossParameter{
  optional float alpha = 1 [default = 0.25];
  optional float gamma = 2 [default = 2];
}

layer {
  name: "focal_loss"
  type: "SoftmaxWithFocalLoss"
  bottom: "ip2"
  bottom: "label"
  top: "focal_loss"
  softmax_focal_loss_param {
    alpha: 1 
    gamma: 1
  }
}
```

### Notes


```
Loss = -1/M * sum_t alpha * (1 - p_t) ^ gamma * log(p_t)
```


### Sigmoid Form

Here use softmax instead of sigmoid function.
If you want see how to use sigmoid to implement Focal Loss, please see https://github.com/sciencefans/Focal-Loss to get more information.

# Citation
## Focal-Loss
The paper is available at https://arxiv.org/abs/1708.02002.
## Active Convolution
This repository contains the implementation for the paper Active Convolution: Learning the Shape of Convolution for Image Classification.

The code is based on Caffe and cuDNN(v5)
## Deformable Convolutional Networks
Deformable Convolutional Networks.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1703.06211
@misc{chollet2015keras,
  author = {Chollet, François and others},
  title = {Keras},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fchollet/keras}}
}
