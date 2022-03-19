
# Image_classification

## Introduction

Convolutional neural networks ahve become famous in cmoputer vision ever since *AlexNet* popularized deep convolutional neural networks by winning IamgeNet Challenge: ILSVRC 2012. The general trend has been to make deeper and more complicated networks in roder to achieve higher accuracy. However, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and spped. In many real world applications such as robotics, self-driving, the recognition tasks need to be carried out in a timely fashion on a computationally limited platform. That's inspire me to build some effient convolutional neural networks which can be used on Mobile/portable device. 

## Data/Preprocessing Data

The input data are dermoscopic lesion images in JPEG format.

The training data consists of 10015 images.

```
AKIEC: 327
BCC: 514
BKL: 1099
DF: 115
MEL: 1113
NV: 6705
VASC: 142
```
*Imbalanced Data*

during `fit_generator`, set `class_weight='auto'`

The format of raw data is as follows:

<img src='/img/raw_data.png'>

And the format of label is as follows:

<img src='/img/raw_label.png'>

Directly loading all of the data into memory.

```
def read_img(img_name):
    im = Image.open(img_name).convert('RGB')
    data = np.array(im)
    return data

images = []

for fn in os.listdir('C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input'):
    if fn.endswith('.jpg'):
        fd = os.path.join('C:\\Users\gavin\Desktop\ISIC2018_Task3_Training_Input', fn)
        images.append(read_img(fd))
```

That is so memory consuming, even the most state-of-the art configuration won't have enough memory space to process the data the way I used to do it. Meanwhile, the number of training data is not large enough, Data Augumentation is the next step to achieve.

Firstly, chaning the format of the raw data using `reformat_data.py`.

<img src='/img/new_data.png'>

Then doing data augumentation.

```
data_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    validation_split=0.3)
```

`horizontal_flip`: Randomly flip inputs horizontally.
`zoom_range`: Range for random zoom.
`width_shift_range`: fraction of total width.
`height_shift_range`: fraction of total height.
`rotation_range`: Degree range for random rotations.
`validation_split`: Split the dataset into 70% train and 30% val. It will shuffle the data.

Then achieve ImageGenerator:

```
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)
```


## Model/ Tricks to imporve performance on mobile device


### Depthwise Separable Convolution

Depthwise Separable Convolution is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution called a pointwise convolution. The depthwise convolution applies a single filter to each input channel, the pointwise convolution then applies a ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size. 

A standard convolutional layer takes as input a ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D%20%5Ctimes%20D_%7BF%7D%20%5Ctimes%20M) feature map F and produces a ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D%20%5Ctimes%20D_%7BF%7D%20%5Ctimes%20N) feature map G where ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D) is a spatial width and height of a square input feature map, M is the number of input channels, ![](https://latex.codecogs.com/gif.latex?D_%7BG%7D) is the spatial width and height of a square output feature map and N is the number of output channel.

The standard convolutional layer is parameterized by convolution kernel K of size ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ctimes%20D_%7BK%7D%20%5Ctimes%20M%20%5Ctimes%20N) where ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D) is the spatial dimension of the kernel assumed to be square and M is number of input channels and N is the number of output channels as defined previously. 

Standard convolutions have the computational cost of:


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)

where the computational cost depends multiplicatively on the number of input channels M, the number of output channels N, the kernel size ![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ctimes%20D_%7BK%7D) and the feature map size ![](https://latex.codecogs.com/gif.latex?D_%7BF%7D%20%5Ctimes%20D_%7BF%7D).

The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation. The filtering and combination steps can be split into two steps via the use of factorized convolutions called depthwise separable convolutions for substantial reduction in computational cost.

Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. Using depthwise convolutions to apply a single filter per input channel (input depth). Pointwise convolution, a simple ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution, is then used to create a linear combination of the output of the depthwise layer.


<img src = '/img/depthwise separable convolution.png'>


Depthwise convolution has a computational cost of


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)


Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolution is needed in order to generate these new features.

The combination of depthwise convolution and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) (pointwise) convolution is called depthwise separable convolution which was originally introduced in.

Depthwise separable convolutions cost:


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%20&plus;%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)


which is the sum of the depthwise and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) pointwise convolutions.

By expressing convolution as two step process of filtering and combining, there's a reduction in computation of


![](https://latex.codecogs.com/gif.latex?%5Cfrac%7BD_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%20&plus;%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%7D%7BD_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20M%20%5Ccdot%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20&plus;%20%5Cfrac%7B1%7D%7BD%5E2_%7BK%7D%7D)

Although using Depthwise Separable Convolution make the model already small and low latency, many times a specific use case or application may require the model to be smaller and faster. In order to construct these smaller and less computationally expensive models, I also set two hyper-parameters: width multiplier and resolution multiplier. The role of the width multiplier ![](https://latex.codecogs.com/gif.latex?%5Calpha) is to thin a network uniformly at each layer. For a give layer and width multiplier ![](https://latex.codecogs.com/gif.latex?%5Calpha), the number of input channels M becomes ![](https://latex.codecogs.com/gif.latex?%5Calpha)M and the number of output channels N becomes ![](https://latex.codecogs.com/gif.latex?%5Calpha)N.

The computational cost of a depthwise separable convolution with width multiplier ![](https://latex.codecogs.com/gif.latex?%5Calpha) is:


![](https://latex.codecogs.com/gif.latex?D_%7BK%7D%20%5Ccdot%20D_%7BK%7D%20%5Ccdot%20%5Calpha%20M%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D%20&plus;%20%5Calpha%20M%20%5Ccdot%20%5Calpha%20N%20%5Ccdot%20D_%7BF%7D%20%5Ccdot%20D_%7BF%7D)

where ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%5Cin%20%280%2C%201%5D) witth typical settings of 1, 0.75, 0.5 and 0.25. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%5E2). 

**The structure of the model**


| Type/Strike |  Filter shape | Input Size |
| ------ | ------ | ------ | 
| Conv2D/s=2 | 3 * 3 * 32 | 224 * 224 * 3 |
| DWConv2D/s=1 | 3 * 3 * 32dw | 112 * 112 * 32 |
| Conv2D/s=1 | 1 * 1 * 32 * 64 | 112 * 112 * 32 |
| DWConv2D/s=2 | 3 * 3 * 64dw | 112 * 112 * 64 |
| Conv2D/s=1 | 1 * 1 * 64 * 128 | 56 * 56 * 64 |
| DWConv2D/s=1 | 3 * 3 * 128dw | 56 * 56 * 128 |
| Conv2D/s=1 | 1 * 1 * 128 * 128 | 56 * 56 * 128 |
| DWConv2D/s=2 | 3 * 3 * 128dw | 56 * 56 * 128 |
| Conv2D/s=1 | 1 * 1 * 128 * 256 | 28 * 28 * 256 |
| DWConv2D/s=1 | 3 * 3 * 256dw | 28 * 28 * 256 |
| Conv2D/s=1 | 1 * 1 * 256 * 256 | 28 * 28 * 256 |
| DWConv2D/s=2 | 3 * 3 * 256dw | 28 * 28 * 256 |
| Conv2D/s=1 | 1 * 1 * 256 * 512 | 14 * 14 * 256 |
| 5 * DWConv2D, Conv2D/s=1 | 3 * 3 * 512dw, 1 * 1 * 512 * 512 | 14 * 14 * 512 |
| DWConv2D/s=2 | 3 * 3 * 512dw | 14 * 14 * 512 |
| Conv2D/s=1 | 1 * 1 * 512 * 1024 | 7 * 7 * 512 |
| DWConv2D/s=1 | 3 * 3 * 1024dw | 7 * 7 * 1024 |
| Conv2D/s=1 | 1 * 1 * 1024 * 1024 | 7 * 7 * 1024 |
| Pooling/FC |  |  |


#### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=32, lr=1, epochs=100 | 2:02:35.984546 | loss: 0.0496 - acc: 0.9844 - val_loss: 0.9146 - val_acc: 0.8437 |
| batch_size=32, lr=0.1, epochs=100 | 2:02:24.674883 | loss: 0.0167 - acc: 0.9938 - val_loss: 0.8037 - val_acc: 0.8592 |
| batch_size=32, lr=0.01, epochs=100 | 2:00:44.432700 | loss: 0.1524 - acc: 0.9481 - val_loss: 0.5204 - val_acc: 0.8420 |
| batch_size=32, lr=0.001, epochs=100 | 1:59:59.366265 | loss: 0.5579 - acc: 0.8000 - val_loss: 0.6101 - val_acc: 0.7804 |
| batch_size=16, lr=0.1, epochs=100 | 2:11:34.221687 | loss: 0.0326 - acc: 0.9896 - val_loss: 0.8072 - val_acc: 0.8539 |
| batch_size=16, lr=0.01, epochs=100 | 2:13:38.293478 | loss: 0.1412 - acc: 0.9524 - val_loss: 0.5066 - val_acc: 0.8372 |
| batch_size=8, lr=0.1, epochs=100 | 2:29:13.814567 | loss: 0.0488 - acc: 0.9836 - val_loss: 0.8756 - val_acc: 0.8473 |
| batch_size=32, lr=0.1, epochs=100, alpha=0.75 | 1:59:59.084560 | loss: 0.0336 - acc: 0.9878 - val_loss: 0.7870 - val_acc: 0.8535 |
| batch_size=32, lr=0.1, epochs=100, alpha=0.5 | 2:00:03.366160 | loss: 0.0554 - acc: 0.9798 - val_loss: 0.6174 - val_acc: 0.8488 |
| batch_size=32, lr=0.1, epochs=100, init_w=xaveir | 2:01:36.719090 | loss: 0.3441 - acc: 0.8718 - val_loss: 0.7564 - val_acc: 0.7605 |





#### Inverted Residuals and Linear Bottlenecks

**Deep Residual Learning for Image Recognition**

Deep convolutional neural networks have led to a series of breakthroughs for image classification. Deep networks naturally integrate low/mid/high-level features and classifiers in an end-to-end multilayer fashion, and the 'levels' of features can be enriched by the number of stacked layers(depth). Evidence reveals that network depth is of crucial importance.

Driven by the significance of depth, a question arises: *Is learning better networks as easy as stacking more layers?*  An obstacle to answering this question was the notorious problem of vanishing/exploding gradients, which hamper convergence from the beginning. When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing ,accuracy gets saturated ( which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error.

<img src='/img/gradient_vanishing.png'>

Someone address the degradation problem by introducing a *deep residual learning* framework. In stead of hoping each few stacked layers directly fit a desired underlying mapping, we can let these layers fit a residual mapping. 

<img src='/img/residual block.png'>

Formally, denoting the desired underlying mapping as ![](https://latex.codecogs.com/gif.latex?H%28x%29), it let the stacked nonlinear layers fit another mapping of ![](https://latex.codecogs.com/gif.latex?F%28X%29%3A%3DH%28x%29%20-%20x). The original mapping is recast into ![](https://latex.codecogs.com/gif.latex?F%28x%29%20&plus;%20x). The formulation of ![](https://latex.codecogs.com/gif.latex?F%28x%29%20&plus;%20x) can be realized by feedforward neural networks with 'shortcut connections' are those skipping one or more layers, the shortcut connections simply perform *identity* mapping, and their outputs are added to the outputs of the stacked layers. Identity shortcut connections add neither extra parameter nor computational complexity.

### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=8, lr=1, epochs=100 | 2:47:20.486248 |loss: 0.3449 - acc: 0.8718 - val_loss: 1.1156 - val_acc: 0.7741|
| batch_size=8, lr=0.01, epochs=200 | 5:41:32.741626 |loss: 0.0190 - acc: 0.9937 - val_loss: 1.1468 - val_acc: 0.8460|


**Deeper Bottleneck Architectures**

<img src='/img/bottleneck.png'>

For each residual function ![](https://latex.codecogs.com/gif.latex?F), using a stack of 3 layers instead of 2. The three layers are ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201), ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203), and ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolutions, where the ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) layers are responsible for reducing and then increasing(restoring) dimensions, leaving the ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) layer a bottleneck with smaller input\output dimensions.

**Linear Bottlenecks**


Comparing of Depthwise Separable Convolution and Linear Bottleneck.

<img src='/img/comparing of mobilenet v1_v2.png'>

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.

Using linear layers is crucial as it prevents non-linearities from destroying too much information.

<img src='/img/linear bottleneck.png'>

Examples of ReLU transformations of low-dimensional manifolds embedded in higher-dimensional spaces. In these examples the initial spiral is embedded into an n-dimensional space using random matrix T followed by ReLU, and then projected back to the 2D space using $T^{-1}$. In examples above n = 2,3 result in information loss where certain points of the manifold collapse into each other, while for n=15 to 30 the transformation is highly non-convex.

**Inverted residuals**


The inverted design is considerably more memory efficient.

<img src='/img/inverted block.png'>

Comparing of bottleneck and inverted residuals.

<img src='/img/comparing of bottleneck.png'>

**basic implementation structure**

Bottleneck residual block transforming from k to k' channels, with stride = s and expansion factor t.

|  Input | Operator | Output |
| ------ | ------ | ------ |
| h * w * k| 1 * 1 conv2d, ReLU6 | h * w * tk |
| h * w * tk | 3 * 3 dw, s=s, ReLU6 | h/s * w/s * tk |
| h/s * w/s * tk | linear 1 * 1 conv2d | h/s * w/s * k' |


**Structure of the model**

| Input | Operator | expansion | output channels | t | s |
| :-: | :-: | :-: | :-: | :-: | :-: | 
| 224^2 * 3   | Conv2D | - | 32     | 1      |    2   |
| 112^2 * 32  | bottleneck | 1 | 16 | 1      |    1   |
| 112^2 * 16  | bottleneck | 6 | 24 | 2      |    2   |
| 56^2 * 24   | bottleneck | 6 | 32 | 3      |    2   |
| 28^2 * 32   | bottleneck | 6 | 64 | 4      |    2   |
| 14^2 * 64   | bottleneck | 6 | 96 | 3      |    1   |
| 14^2 * 96   | bottleneck | 6 | 160 | 3     |    2   |
| 7^2 * 160   | bottleneck | 6 | 320 | 1     |    1   |
| 7^2 * 320   | conv2d 1 * 1 | - | 1280 | 1     |      |
| 7^2 * 1280  | avgpool/FC | - |  |     |    -   |


#### Result

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=32, lr=1, epochs=100 | 2:01:26.619140 |loss: 0.0795 - acc: 0.9733 - val_loss: 1.6232 - val_acc: 0.7831|
| batch_size=32, lr=0.1, epochs=100 | 2:01:06.484069 |loss: 0.0183 - acc: 0.9941 - val_loss: 1.1009 - val_acc: 0.8404|
| batch_size=32, lr=0.01, epochs=100 | 2:01:09.871235 |loss: 0.1427 - acc: 0.9521 - val_loss: 0.5747 - val_acc: 0.8397|


#### Channel shuffle for Group Convolution

Modern convolutional neural networks usually consist of repeated building blocks with the same structure, such as *Xception* and *ResNeXt* introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, both designs do not fully take the ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolutions into account, which require considerable complexity. For example, in ResNeXt only ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) layers are equipped with group convolutions. As a result, for each residual unit in ResNeXt the pointwise convolutions occupy 93.4% multiplication-adds( cardinality = 32 as suggested in). In tiny networks, expensive pointwise convolutions result in limited number of channels to meet the complexity constraint, which might significantly damage the accuracy.

To address the issue, a straightforward solution is to apply channel sparse connections, for example group convolutions, also on ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) layers.By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels. It is clear that outputs from a certain group only relate to the inputs within the group. This property blocks information flow between channel groups and weakens representation

If we allow group convolution to obtain input data from different groups , the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups. 

<img src='/img/channel_shuffle.png'>




#### ShuffleNet Unit


Starting from the design principle of bottleneck unit. In its residual brach, for the ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) layers, applying a computational economical ![](https://latex.codecogs.com/gif.latex?3%20%5Ctimes%203) depthwise convolution on the bottleneck feature map. Then, replacing the first ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) layer with pointwise group convolution followed by a channel shuffle operation. The purpose of the second pointwise group convolution is to recover the channel dimension to match the shortcut path. For simplicity, no apply an extra channel shuffle operation after the second pointwise layer as it results in comparable scores. The usage of batch normalization(BN) and nonlinearity is similar, except not use ReLU after depthwise convolution as suggested by *Xception*. 

<img src='/img/shufflenet_unit.png'>

**Sturcture of Model**

|  Layer | Output size | Ksize | Stride | Repeat | Output channels(8 groups)|
| :-: | :-: | :-: | :-: | :-: | :-: |
|  Image | 224^2  |        |        |        |    3   |
|  Conv1<br>Pool | 112^2<br>56^2  | 3 * 3<br>3 * 3 |    2<br>2    |    1    |    24   |
|  Stage2        | 28^2<br>28^2   |                |    2<br>1    |  1<br>3 |  384<br>384 |
|  Stage3        | 14^2<br>14^2   |                |    2<br>1    |  1<br>7 |  768<br>768 |
|  Stage4        | 7^2<br>7^2     |                |    2<br>1    |  1<br>3 |  1536<br>1536 |
|  GlobalPool/FC        |    |                |        |   |   |


**Result**

|  values of parameters | training time | result |
| ------ | ------ | ------ |
| batch_size=16, lr=1, epochs=500 | 9:59:17.423318 | loss: 0.4489 - acc: 0.8302 - val_loss: 0.6918 - val_acc: 0.7716|
| batch_size=16, lr=0.1, epochs=200 | 4:40:04.491828 |loss: 0.2473 - acc: 0.9078 - val_loss: 0.8477 - val_acc: 0.7829|

## 

https://arxiv.org/pdf/1807.11164.pdf

#### Practical Guidelines for Efficient Network Design

**1. Equal channel width minimized memory access cost(MAC)**

<img src='/img/G1.png'>

Validation experiment for Guideline 1. Four different ratios of number of input/output channels(c1 and c2) are tested, while the total FLOPs under the four ratios is fixed by carying the number of channels.

**2. Excessive group convolution increases MAC**

<img src='/img/G2.png'>

Validation experiment for Guideline 2. Four values of group number g are tested, while the total FLOPs under the four values is fixed by varying the total channel number c.

**3. Network fragmentation reduces degree of parallelism**

<img src='/img/G3.png'>

Validation experiment for Guideline 3. c denotes the number of channels for 1-fragment. The channel number in other fragmented structures is adjusted so taht the FLOPs is the same as 1-fragment.

** 4.Element-wise operations are non-negligible**

<img src='/img/G4.png'>

Validation experiment for Guideline 4. The ReLU and shortcut operations are removed from the bottleneck unit, separately. c is the number of channels in unit.

**Conclusion and Discussions**

1. use 'balanced' convolutions( equal channel width);

2. be aware of the cost of using group convolution;

3. reduce the degree of fragmentation;

4. reduce element-wise operations.

**Channel Split**

<img src='/img/shufflenet_v2.png'>

At the begining of each unit, the input of c feature channels are split into two braches with c - c' and c' channels, respectively. Following G3, one branch remains as identity. The other branch consist of three convolutions with the same input and output channels to satisfy G1. The two ![](https://latex.codecogs.com/gif.latex?1%20%5Ctimes%201) convolutions are no longer group-wise. This is partially to follow G2, and partially because the split operation already produces two groups.

After convolution ,the two branches are concatenated. So, the number of channels keeps the same(G1). The same 'channel shuffle' operation is then used to enable information communication between the two branches. After the shuffling, the next unit begins. Note that the 'Add' operation no longer exists. Element-wise operations like ReLU and depthwise convolutions exist only in one branch. Also, the three successive elementwise operations, 'Concat', 'Channel Shuffle' and 'Channel Split', are merged into a single element-wise operation. These changes are beneficial according to G4.




# Compare Base ResNet50

All models with

```
batch_size = 8
epochs = 100
optimizier = adadelta, lr = 1, decay = 0
```

|  model | Params | result | training time | prediction time(1512 images) |
| :-: | :-: | :-: | :-: | :-: |
| ResNet50 | 23M | 78.45% | 2:46:23 | 00:03:57/0.157s |
| Separable_dw | 12M | 76.83% | 2:27:33 | 00:03:49/0.151s |
| Separable_dw_linear | 12M | 77.47% | 2:24:06 | 00:03:43/0.147s |
| inverted_residual | 2.2M | 77.28% | 2:32:11 | 00:03:35/0.142s |
| shuffle_unit | 0.9M | 78.25% | 2:49:22 | 00:05:58/0.237s |
| shuffle_unit_without_shuffle | 0.9M | 80.22% | 2:46:05 | 00:05:54/0.234s |

All models with

```
batch_size = 8
epochs = 100
optimizier = adadelta, lr = 1, decay = 0.05
```

|  model | Params | result | training time |
| :-: | :-: | :-: | :-: |
| ResNet50 | 23M | 78.52% | 2:47:40 |
| inverted_residual | 2.2M | 77.35% | 2:32:25 |
| shuffle_unit | 0.9M | 75.24% | 2:50:35 |
| shuffle_unit_without_shuffle | 0.9M | 76.24% | 2:47:31 |
