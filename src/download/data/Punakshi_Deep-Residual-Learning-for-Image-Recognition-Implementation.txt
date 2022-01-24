# Deep-Residual-Learning-for-Image-Recognition-Implementation
Based on paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - https://arxiv.org/pdf/1512.03385.pdf

Insights:
1. The residual has been defined as I(x) = H(x) - F(x)

  where I(x) is identity mapping which is I(x) =x,H(x) the desired mapping ,F(x) the mapping which network layers where able to achieve according to their inputs.

  H(x) = F(x) +I(x) is our objective and only F(x) is changing everytime assuming H(x) and I(x) as constants.

  Rather than learning F(x) in a simple deep network which can have degradation problem its helpful for the model to give accurate results. I(x) is adding the deficiencies in F(x) at each skip connection if it is less than H(x) and absorbing all the errors in F(x) is more than H(x).

2. One of the basic principles involved is the idea to build a deeper network by simply repeating this module i.e the smaller network.

3.A standard two layer module is as below:

def Unit(x,filters):

  out = BatchNormalization()(x)
  
  out = Activation("relu")(out)
  
  out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
  
  
  out = BatchNormalization()(out)
  
  out = Activation("relu")(out)
  
  out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
  
  return out
  
  Whereas a resNet module is:  
    
def Unit(x,filters):

  res = x
  
  out = BatchNormalization()(x)
  
  out = Activation("relu")(out)
  
  out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
  
    
  out = BatchNormalization()(out)
  
  out = Activation("relu")(out)
  
  out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
  
  
  out = keras.layers.add([res,out])
  
  
  return out
  

Firstly store a reference “res” to the original input, and after passing through the batchnorm-relu-conv layers,  add the output to the residual.This part corresponds to the equation y = f(x) + x instead of standard network y = f(x) 

4. The key insight of this paper is that the main power of deep residual networks is in residual blocks, and that the effect of depth is actually supplementary (contrary to what was believed).

5. Related Study: ResNeXt, Analysis of claims:  (1) Residual Nets being equivalent to RNN and (2) Residuals Nets acting more like ensembles across several layers, Similarities between LSTM and Highway Networks with Residual Network.
