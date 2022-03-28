# BlurPool
Caffe implementation of Adobe paper: "Making Convolutional Networks Shift-Invariant Again"

****
The Origin Paper : <https://arxiv.org/abs/1904.11486>  
Project page : <https://richzhang.github.io/antialiased-cnns>   
Ptorch Version : <https://github.com/adobe/antialiased-cnns> 

****

To be convenient, this repository efficiently adds the blur kernel effect on the base_conv_layer by modifying the LayerSetup.  
By directly rewrite the base_conv_layer, it will not change the layer type(convolution).  
It only needs to add the convolutional param: blur_kernel  
But the drawback is that Caffe uses zero paddings instead of other types.  
```
**Note that this is same as the origin paper which using depthwise**
**Here will also check the bias_term (suppose to be false) and the lr_rate, decay_rate should be 0**
```
****
caffe.proto  
```caffe
message ConvolutionParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms
  
  ......
  
  optional bool blur_kernel = 18 [default = false];
}

```  

Caffe protoxt  
```caffe
layer {
  name: "Conv_BlurPool"
  type: "Convolution"
  bottom: "Conv1"
  top: "Conv_BlurPool"
  param {
    lr_mult: 0.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 1
    kernel_size: 3
    group: 64
    stride: 2
    blur_kernel: true
  }
}
```
