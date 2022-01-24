# Deep Learning with PyTorch Notes

Source: [https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

Notes made for chosen chapters when reading a book. 
Additional informations are included in each jupyter notebook for given chapter.

## Chapter 5

- 110 - L1 vs L2 metric for error (MSE)
- 111 - torch.ones(()), using size, shape, type functions, broadcasting (more on broadcasting semantics: [https://pytorch.org/docs/stable/notes/broadcasting.html](https://pytorch.org/docs/stable/notes/broadcasting.html), examples in Jupyter).
- 113 - idea of a gradient descent, rate of change of the function (ROC) formula
- 114 - why the analytically calcuated gradient is better then calculating ROC for discrete values
- 118 - overtraining with too big learning rate problem
- 119 - input normalization, problem of having parameters of different scale

having different learning rates for different parameters is not a good option, becomes redundant as the model scales up. always normalize the inputs.

- 123 - using autograd functionality

calling backward on a function accumulates the gradient. after updating parameters use **params.grad.zero_()** to set the gradient to 0 before next iteration of gradient calculating.

usage of **torch.no_grad()**: it stops tracing the grad changes. used when updating the parameters, since grad would track that in-place change being made.
another way of doing that: creating a copy of a tensor using the **tensor.detach()** functionality (also removes the grad tracking).
also **torch.set_grad_enabled()** which takes boolean value could be used

- 127 - adding the optimizers, torch.optim

optimizers have two primary methods: **zero_grad** and **step.** when constructing optimizer, we are passing the parameters inside. the **zero_grad** will zero the grad for these parameters. **step** method is updating the value of the parameters (we don't have to change the values of parameters after backward pass in place anymore)

- 130 - ADAM optimizer

ADAM is a lot less sensitive to the scale of the parameters than SGD. it has the adaptive learning rate property.

- 131 - train/test/valid dataset separation, overfitting

if the training loss and validation loss are diverge with the training going forward, it means we are overfitting. the model is fitting too well to the training data and therefore it's not generalizing well

ways to prevent overfitting:
1. using more data (more samples for model to learn → statistically better generalizing)
2. using regularization methods - penalizing big differences between the prediction and ground truths. this makes sure that the model will be as regular in-between the training data.
3. using data augmentation - providing more training samples by manipulation of already existing data (adding noise, translations etc.)
4. making the model simpler - having less parameters → a bit worse fit to the data → less error during prediction on new data
5. apply dropout
usual way of performing the neural network fitting in two steps:
1. make it bigger until it fits the data
2. scale it down so it prevents overfitting.

- 134 - splitting a dataset

**torch.randperm()** is shuffling the tensor indices

## Chapter 6

- 143 - artificial neuron, weight/bias
- 144 - multilayer network
- 145 - activation function

activation function has two main roles:
1. it presents non-linearity to the model. normally, neuron construction only allows linear changes (mul and add in the neuron). we can later apply activation function of any type (tanh, relu etc.) to change the output.
2. it allows us to get a certain type of output from the network when included on the last layer (for instance classes with softmax)

- 147 - types of activation functions

ReLU - widely used, one of the state-of-the-art performances in networks.
Softmax - used to be best, now used mostly when the output from the network should be within the [0,1] range (e.g. probability)

activation functions properties:
1. non-linear
2. differantiable (for gradient calculation). point discontinuities are fine (so ReLU works)

- 151 - torch.nn module
- 152 - __call__ vs forward in nn.Module
- 155 - batching

reasons for batching:
1. processing only one sample at the time doesn't saturate CPU/GPU performance. these are normally parallelized which means we can perform whole batches at once to use all of the computing power
2. some models use statistics which are calculated on the whole batches

to use the nn.Module we need to provide the data where first dimension is the batch quantity. for single tensors as inputs we can use **torch.unsqueeze()**

- 159 - nn.Sequential
- 160 - inspecting the parameters of the network

use **model.parameters()** (it's a generator) to inspect the parameters.
to check them with names use **model.named_parameters().**

use OrderedDict to define sequential model with layer names. makes it easier for parameters inspection later.

## Chapter 7

- 165 - torchvision, downloading dataset
- 168 - torchvision.transforms

**torchvision.transforms** provides a translation of PIL and numpy to tensors. it can be used directly when loading a dataset when using **transform=transforms.ToTensor()**

matplotlib expects HxWxC shape (color channels at the end) when plotting. pytorch uses CxHxW so there is a need to **permute** it.

- 170 - normalizing the data, using **transform.Compose**

**torch.view** allows to make calculations on a reshaped tensor without defining a new one (same storage is used), works similar to **np.reshape** (https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)

- 175 - classification problem

dealing with the categorical input requires using different functions. **softmax** gives us output as a probability at the end of the network

- 180 - loss functions for classification, NLLLoss

negative log-likelihood (in theory):
1. for a single sample in the batch, look at the probability value *p* that model assigned to the correct class.
2. calculate the -log(*p*), which is high when probability is low and vice versa
3. add that to the overall loss for a batch

in pytorch, the NLLLoss is not implemented as in theory, it doesn't apply the log value to the probability. therefore, we need to already input the logarithmic value inside of the loss function. for that, we should switch **softmax** to **logsoftmax** in the model, so the output tensor from the model is already with log.

- 183 - sample, mini-batch and batch training
- 185 - DataLoader

using **LogSoftmax** with **NLLLoss** results in the same result as using **nn.CrossEntropyLoss** at one go for the loss value (without using any softmax function at the end of the network). difference  is, the results of the network won't be presentable as probabilities (or log_probabilities), since softmax gives us that.

- 186 - checking the accuracy of the model
- 189 - limitations of nn.Linear

fully connected network is not translation invariant. recognizing a pattern in one place on the image doesn't contribute to recognizing the pattern in a different place. that is why linear layers are redundant - they need too many parameters to take every possibility into account.

## Chapter 8

- 194 - convolutions

using convolution kernels instead of fully-connected layers:
1. local operations on the neighboourhoods,
2. translation invariance (we slide the kernel over the whole image)
3. fewer parameters

- 198 - padding

padding is used to leave the image dimensions unchanged after the convolution layer. important in big architectures.

- 203 - downsampling

idea of downsampling: even though convolutions are reducing the number of parameters and equip us with locality and translation invariance, the kernels have limited (and usually *small*) size. therefore we are unable to get the "big picture" properties of an image. to address that, we could use bigger kernels or use downsampling between convolutions.

downsampling techniques:
1. average pixels - e.g. 4 pixels from each 2x2 square are being represented as one using the average value of all of them
2. max pixels - from 4 pixels we leave one with the highest value (the most responsive one) (**nn.MaxPool2d**)
3. strided convolution when we are looking with the convolutional filter at the pixels that are further from each other.

- 205 - receptive field of an output neuron

use **numel()** to get the number of parameters in each of the network layer

when working with **nn.Sequential** we can get from multichannel, 2d layer output (for instance after conv2d layer) to the Linear layer (at the end, when we want to calculate the probabilities) using the **nn.Flatten** functionality. without it, we must build the forward network pass by hand.

- 208 - building **nn.Module** subclass and defining forward pass
- 210 - using **nn.functional**
- 211 - functional vs modular API

main difference between using **nn.Module** and **nn.functional** utilities is that with the former, we trace the parameters, the latter is purely functional input → output mapping, nothing is remember. useful e.g for activation functions, pooling etc.

- 214 - saving a model
- 215 - loading a model
- 215 - GPU training

when loading model that was trained on different device we want to load it into, we can use **map_location** keyword in **torch.load** function.

- 218 - adding width to the network
- 219 - regularization methods

L2 regularization is called *weight decay*, it sums the squares of the weights in the model
L1 is called *lasso regression,* it is a sum of absolute values in the model
L1 is good for sparsity, when there are many inputs and you believe that only a few of them are meaningful. L2 is good at dealing with correlated inputs. If two inputs are perfectly correlated L2 put half of the weight (beta) on each input, while L1 would pick one randomly (so less stable). One can use a combination of L1, L2 to get a balance of both, also known as Elastic Net.

- 221 - dropout
- 222 - batch normalization

reckon that when using **nn.BatchNorm2d** in the network we should set bias=False in the **nn.Conv2d** since the bias is already included in the batch normalization part. 

- 223 - **model.train()** and **model.eval()**
- 224 - vanishing gradient problem

deep networks prior to 2015 had ~20 layers, the training on more of them was highly uneffective due to the vanishing gradient problem. Multiplying a lot of small numbers during backpropagation led to the parameters on the early layer to be very difficult to train properly. problem was addressed in 2015 by introducing the ResNets (https://arxiv.org/abs/1512.03385) and reaching even 100 layers in the networks.

- 227 - creating deep ResNet sequentially with the usage of ResBlocks.
- 228 - weight initialization
- 230 - overgeneralization

## Chapter 10

- 257 - bash commands to explore the .csv data.

use **wl -l** command to count the number of rows, use **head** to check the first few lines

- 259 - collections.namedtuple usage
- 260 - caching data on the disk
- 263 - MetalIO format from SimpleITK
- 271 - custom Dataset creation

 Dataset subclasses in PyTorch API need two methods: "__len__" and "__getitem__"

- 275 - training/validation datasets

both of validation and training sets should have a good representation of all of the data (all variations of inputs). unless the training is meant to be robust for outliers they should not include any.
