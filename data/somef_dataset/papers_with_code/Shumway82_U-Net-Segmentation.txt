# U-Net Segmentation

"U-Net: Convolutional Networks for Biomedical Image Segmentation"

Paper: https://arxiv.org/pdf/1505.04597.pdf

#### Abstract
There is large consent that successful training of deep networks
requires many thousand annotated training samples. In this paper,
we present a network and training strategy that relies on the strong
use of data augmentation to use the available annotated samples more
efficiently. The architecture consists of a contracting path to capture
context and a symmetric expanding path that enables precise localization.
We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window
convolutional network) on the ISBI challenge for segmentation of neuronal
structures in electron microscopic stacks. Using the same network
trained on transmitted light microscopy images (phase contrast
and DIC) we won the ISBI cell tracking challenge 2015 in these categories
by a large margin. Moreover, the network is fast. Segmentation
of a 512x512 image takes less than a second on a recent GPU. The full
implementation (based on Caffe) and the trained networks are available
at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.

![bad](http://deeplearning.net/tutorial/_images/unet.jpg)
#### Figure 1: U-Net architecture

## Challenge

There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. The training set contains of 25 labeled images and the test set is unlabeled. The task is to train a neural network which predicts the roofs of the house based on the training sets. 

## Approuch

To solve this task, a U-Net architecture was chosen. The architecture was shown in Figure 1. Since very few training images exist, data augmentation was used. For this, the image was randomly flipped horizontally and vertically and randomly rotated at an angle of one degree. As loss-function tree different loss functions, the dice-loss and pixelwise-softmax and the softmax of tensorflow, are used and compared. 

## Results

The results are produced under the following conditions:

Hyperparameter: Batch-Size = 8, instance normalization, Cyclic-Learningrate [1e-4,..., 1e-5], epochs = 10000, Data augumentation = flipping [horizontal, vertical] and rotation [0,...,360], Adam with beta1 = 0.9, Filter size of first conv-layer = 64, Unet-depth = 5 

#### DICE

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/mask_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/image_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/535.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/mask_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/image_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/537.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/mask_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/image_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/539.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/mask_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/image_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/551.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/mask_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/dice/image_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/553.png)
#### Reference: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

#### Pixelwise-Softmax

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/mask_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/image_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/535.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/mask_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/image_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/537.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/mask_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/image_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/539.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/mask_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/image_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/551.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/mask_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/pixel/image_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/553.png)
#### Reference: https://arxiv.org/pdf/1505.04597.pdf

#### Softmax

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/mask_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/image_000000.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/535.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/mask_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/image_000001.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/537.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/mask_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/image_000002.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/539.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/mask_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/image_000003.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/551.png)

![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/mask_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/predictions/softmax/image_000004.png)
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/test_X/553.png)

#### Accuracy
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/images/accuracy.png)
##### Accuracy for train(left) and test-set(right) 

#### Intersection of Union IOU
![bad](https://github.com/Shumway82/U-Net-Segmentation/blob/master/Data/images/iou.png)
##### Intersection of Union(IOU) for train(left) and test-set(right) 

## Installation tf_base package
1. Clone the repository
```
$ git clone https://github.com/Shumway82/tf_base.git
```
2. Go to folder
```
$ cd tf_base
```
3. Install with pip3
``` 
$ pip3 install -e .
```

## Install U-Net-Segmentation package

1. Clone the repository
```
$ https://github.com/Shumway82/U-Net-Segmentation.git
```
2. Go to folder
```
$ cd Binary-Classification
```
3. Install with pip3
```
$ pip3 install -e .
```

## Usage-Example

1. Training
```
$ python pipeline_trainer.py --dataset "../Data/" --loss "dice"
```

2. Inferencing
```
$ python pipeline_inferencer.py --dataset "../Data/" --model_dir "Models_dice" 
```
