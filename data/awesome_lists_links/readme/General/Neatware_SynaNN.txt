

#                       SynaNN: A Synaptic Neural Network 



## 1. Introduction

Synapses play an important role in biological neural networks.  They're joint points of neurons where learning and memory happened. The picture below demonstrates that two neurons (red) connected through a branch chain of synapses which may  link to other neurons. 

<p align='center'>
<img src="./picture/synapse.jpg" alt="synapse" width="80%" />
</p>

Inspired by the synapse research of neuroscience, we construct a simple model that can describe some key properties of a synapse. 

<p align='center'>
<img src="./picture/synapse-unit.png" alt="synpase" width="70%" /> 
</p>

A Synaptic Neural Network (SynaNN) contains non-linear synapse networks that connect to neurons. A synapse consists of an input from the excitatory-channel, an input from the inhibitory-channel, and an output channel which sends a value to other synapses or neurons. The synapse function is

<p align='center'>
<img src="https://latex.codecogs.com/svg.latex?S(x,y;\alpha,\beta)=\alpha%20x(1-\beta%20y)"
</p>

where x∈(0,1) is the open probability of all excitatory channels and α >0 is the parameter of the excitatory channels; y∈(0,1) is the open probability of all inhibitory channels and β∈(0,1) is the parameter of the inhibitory channels. The surface of the synapse function is  

<p align='center'>
<img src="./picture/synpase.png" alt="synpase" width="50%" />
</p>

By combining deep learning, we expect to build ultra large scale neural networks to solve real-world AI problems. At the same time, we want to create an explainable neural network model to better understand what an AI model doing instead of a black box solution.

<p align='center'>
<img src="./picture/E425.tmp.png" alt="synpase" width="60%" />
</p>

A synapse graph is a connection of synapses. In particular, a synapse tensor is fully connected synapses from input neurons to output neurons with some hidden layers. Synapse learning can work with gradient descent and backpropagation algorithms. SynaNN can be applied to construct MLP, CNN, and RNN models.

Assume that the total number of input of the synapse graph equals the total number of outputs, the fully-connected synapse graph is defined as 

<p align='center'>
<img src="https://latex.codecogs.com/svg.latex?y_{i}(\textbf{x};%20\pmb\beta_i)%20=%20\alpha_i%20x_{i}{\prod_{j=1}^{n}(1-\beta_{ij}x_{j})},\%20for\%20all\%20i%20\in%20[1,n]"/>
</p>

where 

<p align='center'>
<img src="https://latex.codecogs.com/svg.latex?\textbf{x}=(x_1,\cdots,x_n),\textbf{y}=(y_1,\cdots,y_n),x_i,y_i\in(0,1),\alpha_i \geq 1,\beta_{ij}\in(0,1))"/>
</p>

Transformed to tensor/matrix representation, we have the synapse log formula, 

<p align='center'>
<img src="https://latex.codecogs.com/svg.latex?log(\textbf{y})=log(\textbf{x})+{\textbf{1}_{|x|}}*log(\textbf{1}_{|\beta|}-diag(\textbf{x})*\pmb{\beta}^T)"/>
</p>

We are going to implement this formula for fully-connected synapse network with Tensorflow and PyTorch in the examples.

Moreover, we can design synapse graph like circuit below for some special applications. 

<p align='center'>
<img src="./picture/synapse-flip.png" alt="synapse-flip" width="50%" />
</p>

## 2. SynaNN Key Features

* Synapses are joint points of neurons with electronic and chemical functions, location of learning and memory

* A synapse function is nonlinear, log concavity, infinite derivative in surprisal space (negative log space)

* Surprisal synapse is Bose-Einstein distribution with coefficient as negative chemical potential

* SynaNN graph & tensor, surprisal space, commutative diagram, topological conjugacy, backpropagation algorithm

* SynaNN for MLP, CNN, RNN are models for various neural network architecture

* Synapse block can be embedded into other neural network models

* Swap equation links between swap and odds ratio for healthcare, fin-tech, and insurance applications

  

## 3. A SynaNN for MNIST by Tensoflow 2.x

Tensorflow 2 is an open source machine learning framework with Keras included. TPU is the tensor processor unit that can accelerate the computing of neural networks with multiple cores and clusters.

MNIST is a data sets for hand-written digit recognition in machine learning. It is split into three parts: 60,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).

By using Synapse layer and simple multiple layers of CNN (Conv2D), MaxPooling, Layer, Activation, Droupout, and Adam for optimization, we achieved very good **99.59%** accuracy . 



### 3.1 Import Tensorflow and Keras

```python
# foundation import
import os, datetime
import numpy as np

# tensorflow import
import tensorflow as tf
import tensorflow_datasets as tfds

# keras import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, Layer, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

# keras accessory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

# ploting
import matplotlib.pyplot as plt
```

These are imports for later use. We are going to apply tensorflow, keras, numpy, and matplotlib.

 

### 3.2 Initialize TPU

```python
# use TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
```

This code clip is for TPU using in colab environment.  Below is the output of TPU initialization.

```
INFO:tensorflow:Initializing the TPU system: grpc://10.116.65.130:8470
INFO:tensorflow:Initializing the TPU system: grpc://10.116.65.130:8470
INFO:tensorflow:Clearing out eager caches
INFO:tensorflow:Clearing out eager caches
INFO:tensorflow:Finished initializing TPU system.
INFO:tensorflow:Finished initializing TPU system.
WARNING:absl:`tf.distribute.experimental.TPUStrategy` is deprecated, please use  the non experimental symbol `tf.distribute.TPUStrategy` instead.
INFO:tensorflow:Found TPU system:
INFO:tensorflow:Found TPU system:
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
INFO:tensorflow:*** Available Device: 
......
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
```



### 3.3 Define Plotting Program

```python
# plot accuracy graph
def plotAccuracy20(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  plt.tight_layout()
```

This is the procedure to draw the accuracy graph.



### 3.4 Define Global Parameters

```python
# global training data
batch_size = 128*4
num_classes = 10
epochs = 35
hidden_size = 196*4 
```

Define batch size, epochs, and hidden_size.



### 3.5 Define Synapse Class as a Layer

```python
 class Synapse(Layer):
  # output_dim is the number of output of Synapse
  def __init__(self, output_dim, name=None, **kwargs):
    super(Synapse, self).__init__(name=name)
    self.output_dim = output_dim
    super(Synapse, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    initializer = tf.keras.initializers.RandomUniform(minval=-0.00, maxval=0.01, seed=3)
    config = initializer.get_config()
    initializer = initializer.from_config(config)
    
    # Define kernel
    self.kernel = self.add_weight(name='kernel', 
                                  shape=(input_shape[1], self.output_dim), 
                                  trainable=True)
	# Build Synapse
    super(Synapse, self).build(input_shape)

  # synapse kernel implementation. read the reference paper for explaination.
  def syna_block(self, xx):
    ww2 = self.kernel
    shapex = tf.reshape(tf.linalg.diag(xx), [-1, self.output_dim])
    betax = tf.math.log1p(-tf.matmul(shapex, ww2))
    row = tf.shape(betax)[0]
    allone = tf.ones([row//(self.output_dim), row], tf.float32)
    return xx*tf.exp(tf.tensordot(allone, betax, 1)) #*self.bias

  # call
  def call(self, x):
    return self.syna_block(x)

  # get output shape
  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)
  
  # get config
  def get_config(self):
    config = super(Synapse, self).get_config()
    config.update({'output_dim': self.output_dim})
    return config

```

This is the implementation of Syanapse in Tensorflow. It is a layer to replace Dense in the Keras.



### 3.6 Specify Model

```python
# model
def create_model():
  return Sequential([
       Conv2D(28,  (3, 3), activation='relu', input_shape= (28, 28, 1), trainable=True),
       Conv2D(56,  (3, 3), activation='relu', trainable=True), 
       Conv2D(112, (5, 5), activation='relu', trainable=True),
       Conv2D(hidden_size, (7, 7), activation='relu', trainable=True),
       GlobalMaxPooling2D(),
       Dropout(0.25),
       Flatten(),
       Synapse(hidden_size),
       Dropout(0.25),
       Dense(num_classes)])
```

We created 4 Conv2D as feature extraction along with relu activation. GlobalMaxPooling2D is applied to simplify the Convolution layers. The Synapse layer that implemented SynaNN model is used for fully connected layer. That is the key to classify the images from features. 

 

### 3.7 Define Pre-Processing Dataset

```python
# data pre-processing
def get_dataset(batch_size=64):
  datasets, info = tfds.load(name='mnist', 
                             with_info=True, as_supervised=True, try_gcs=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']
  
  # scale image
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

  # get train and test dataset
  train_dataset = mnist_train.map(scale).shuffle(10000).batch(batch_size)
  test_dataset = mnist_test.map(scale).batch(batch_size)
  return train_dataset, test_dataset
```

This is the pre-processing procedure for machine learning.



### 3.7 Start Training

```python
# get dataset
train_dataset, test_dataset = get_dataset()

# create model and compile
with strategy.scope():
  model = create_model()
  model.compile(optimizer=Adam(lr=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# show model information
model.summary()

# checkpoint setting
checkpoint_path = 'synann_mnist_tpu_model.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_weights_only=False, save_best_only=True)
def lr_sch(epoch):
    if epoch < 12:
        return 1e-3
    if epoch < 30:
        return 1e-4
    if epoch < 65:
        return 1e-5
    if epoch < 90:
        return 1e-6
    return 1e-6
      
# scheduler and reducer setting
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor='accuracy',factor=0.2,patience=5, mode='max',min_lr=1e-5)
callbacks = [checkpoint, lr_scheduler, lr_reducer]

# training start
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset,verbose=1, callbacks=callbacks)

# plot accuracy graph
plotAccuracy20(history)
```

Create model, compile, set checking point, sec scheduler and reducer, start training and plot accuracy graph. The output result with the best accuracy **99.59%** is showed below. The number of iteration is only 31. Excellent!

```text
Epoch 00028: val_accuracy did not improve from 0.99590
938/938 [==============================] - 30s 32ms/step - loss: 0.0035 - accuracy: 0.9990 - val_loss: 0.0225 - val_accuracy: 0.9959
Epoch 29/31
936/938 [============================>.] - ETA: 0s - loss: 0.0027 - accuracy: 0.9992
Epoch 00029: val_accuracy did not improve from 0.99590
938/938 [==============================] - 30s 32ms/step - loss: 0.0027 - accuracy: 0.9992 - val_loss: 0.0258 - val_accuracy: 0.9956
Epoch 30/31
937/938 [============================>.] - ETA: 0s - loss: 0.0026 - accuracy: 0.9992se
Epoch 00030: val_accuracy did not improve from 0.99590
938/938 [==============================] - 29s 31ms/step - loss: 0.0026 - accuracy: 0.9992 - val_loss: 0.0284 - val_accuracy: 0.9954
Epoch 31/31
937/938 [============================>.] - ETA: 0s - loss: 0.0029 - accuracy: 0.9992
Epoch 00031: val_accuracy did not improve from 0.99590
938/938 [==============================] - 29s 31ms/step - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.0265 - val_accuracy: 0.9956

```

<p align='center'>
<img src="./picture/mnist-accuracy.png" alt="synapse-flip" width="50%" />
</p>

### 3.8 Evaluation and Predication

```python
# load model
with strategy.scope():
  new_model=tf.keras.models.load_model(checkpoint_path, custom_objects={'Synapse': Synapse})
  new_model.compile(optimizer=Adam(lr=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
new_model.summary()

# evaluate
loss,acc = new_model.evaluate(test_dataset, verbose=2)
print("Restored model: accuracy = {:5.2f}%".format(100*acc))

# predict
probs = new_model.predict(test_dataset)
print(probs.argmax(axis=1), len(probs))
```

Evaluation and predication.





## 4. A SynaNN for MNIST by PyTorch

PyTroch is an open source machine learning framework that accelerates the path from research prototyping to production deployment.

MNIST is a data sets for hand-written digit recognition in machine learning. It is split into three parts: 60,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).

A hard task to implement SynaNN by PyTorch to solve MNIST problem  is to  define the Synapse class in nn.Module so that we can apply the Synapse module to work with other modules of PyTorch.

The architecture of the codes are divided into header, main, train, test, net, and synapse. 

### 4.1 Header

The header section imports the using libraries. torch, torchvision, and matplotlib are large libraries.

```python
#
# SynaNN for Image Classification with MNIST Dataset by PyTorch
# Copyright (c) 2020, Chang LI. All rights reserved. MIT License.
#
from __future__ import print_function

import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module

import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

train_losses = train_counter = test_counter = test_losses = []
```

### 4.2 Synapse Class

Here is the default API specification of a class in the neural network module of PyTorch. 

```python
class Synapse(nn.Module):
    r"""Applies a synapse function to the incoming data.`

    Args:
        in_features:  size of each input sample
        out_features: size of each output sample
        bias:         if set to ``False``, the layer will not learn an additive bias.
                      Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
             additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
             are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            	:math:`(\text{out\_features}, \text{in\_features})`. The values are
            	initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            	:math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = Synapse(64, 64)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
```

```python
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Synapse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

```

```python
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
```

```python
    # synapse core
    def forward(self, input):
        # shapex = matrix_diag(input)
        diagx = torch.stack(tuple(t.diag() for t in torch.unbind(input,0)))
        shapex = diagx.view(-1, self.out_features)
        betax = torch.log1p(-shapex @ self.weight.t())
        row = betax.size()
        allone = torch.ones(int(row[0]/self.out_features), row[0])
        if torch.cuda.is_available():
          allone = allone.cuda()
        return torch.exp(torch.log(input) + allone @ betax) # + self.bias)    
```

One challenge was to represent the links of synapses as tensors so we can apply the neural network framework such as PyTorch for deep learning. A key step is to construct a Synapse layer so we can embed synapse in deep learning neural network. This has been done by defining a class Synapse.  

```python
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```

### 4.3 Net Class



```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # fully connected with synapse function
        self.fc1 = nn.Linear(320, 64)
        self.fcn = Synapse(64,64)
        self.fcb = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

```

There are two CNN layers for feature retrieving. fc1 is the linear input layer, fcn from Synapse is the hidden layer, and fc2 is the output layer. 

Synapse pluses Batch Normalization can greatly speed up the processing to achieve an accuracy goal. We can think of a synapse as a statistical distribution computing unit while batch normalization makes evolution faster. 

```python
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim=1)
        
        # fcn is the output of synapse
        x = self.fcn(x)
        # fcb is the batch no)rmal 
        x = self.fcb(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

```

### 4.4 Train

```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')

```

### 4.5 Test

```python
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  
```

### 4.6 Main

```python
def main():
    print(torch.version.__version__)
    
    # Training settings
    import easydict
    args = easydict.EasyDict({
      "batch_size": 100,
      "test_batch_size": 100,
      "epochs": 200,
      "lr": 0.012,
      "momentum": 0.5,
      "no_cuda": False,
      "seed": 5,
      "log_interval":100
    })
```



```python
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
   
    device = torch.device("cuda:0" if use_cuda else "cpu")
```

use_cuda is the tag for gpu availability. 

```python
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

```

```python
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
                          
    test_counter = [i*len(train_loader.dataset) for i in range(args.epochs)]
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
```

```python
    # draw curves
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
```

```python
if __name__ == '__main__':
  main()
```

## 5. Results

<p align='center'>
<img src="./picture/synapse-pytorch-99p.jpg" alt="synapse-pytorch-99p" style="width: 80%;" />
</p>

## 6. References

1. ##### SynaNN: A Synaptic Neural Network and Synapse Learning

   https://www.researchgate.net/publication/327433405_SynaNN_A_Synaptic_Neural_Network_and_Synapse_Learning

2. **A Non-linear Synaptic Neural Network Based on Excitation and Inhibition**

   https://www.researchgate.net/publication/320557823_A_Non-linear_Synaptic_Neural_Network_Based_on_Excitation_and_Inhibition

3. 