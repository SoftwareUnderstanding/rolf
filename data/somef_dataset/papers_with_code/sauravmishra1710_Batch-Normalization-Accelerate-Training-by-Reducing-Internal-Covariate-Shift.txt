# Batch Normalization : Internal Covariate Shift

This repository discusses the following two papers on Batch Normalization - 

[1] **Batch Normalization: Accelerate Training by Reducing Internal Covariate Shift**<br>
Paper Link @ https://arxiv.org/abs/1502.03167

[2] **How Does Batch Normalization Help Optimization?**<br>
Paper Link @ https://arxiv.org/abs/1805.11604

A high level paper summary of both the papers are discussed below - 

## Batch Normalization: Accelerate Training by Reducing Internal Covariate Shift
One complication with training deep neural networks is always related to the distribution of inputs. The distribution of input to each layer changes during the training phase since the parameters of the layers change. This phenomenon is known as ***INTERNAL CO-VARIATE SHIFT*** and is known to slow down the learning process. 

**Batch Normalization** is a technique to reduce the internal co-variate shift and improve the training speed, performance and stability of deep neural networks. Batch Normalization make normalization a part of the model architecture and considers normalization for each ***mini-batch*** during the training phase. 

Batch Normalization advantages - 

- Allows us to use much higher learning rates and worry less about initialization. 
- Ocassionally acts as a regularizer, in some cases eliminating the need for Dropout.
- Attempts to solve the vanishing gradient problem.
- Makes the network more stable to the initialization of the weights.

## How Does Batch Normalization Help Optimization?

This paper discusses more in detail of how the batch normalization works. The authors point that input distributions of layer inputs has little to do with the success of BatchNorm.  Instead, a more fundamental impact of Batch Normalization on the training process is that it **makes the optimization landscape significantly smoother**. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training.
