# ILSVRCPlus
Implementations of Deep Convolution Neural Networks for Image classification, train on MNIST, CIFAR-10 and Tiny Imagenet

## Objectives
This repository is dedicated to construct and train various deep architectures used for image classification. More specifically, the goals for this repository are listed below:

* Construct a VGG Network
  * Construct a VGG-like architecture for training on MNIST
  * Construct a VGG-like architecture for training on CIFAR-10
  * Construct the original architecture, as mentioned in [[1]]("1").

* Construct a GoogLeNet Network
  * Construct a GoogLeNet variant for training on MNIST dataset.
  * Construct a GoogLeNet variant for training on the CIFAR-10 dataset.
  * Construct the original architecture, as mentioned in [[2]]("2").

* Implement a ResNet and its variants [[3]]("3"), [[4]]("4")
  
 * Report training and testing results

## To-do list
- [ ] VGGNet
	- [x] VGGNet for CIFAR-10
	- [x] VGGNet for MNIST
	- [ ] VGGNet for Tiny Imagenet

- [ ] GoogLeNet
	- [x] GoogLeNet for CIFAR-10
	- [x] GoogLeNet for MNIST
	- [ ] GoogLeNet for Tiny Imagenet

- [ ] ResNet [[3]]("3"), [[4]]("4")
	- [ ] ResNet for CIFAR-10
	- [x] ResNet for MNIST
	- [ ] ResNet for Tiny Imagenet
	
- [ ] Rank-1 and Rank-5 accuracy

- [ ] Provide the notebooks

## Results
<details>
	<summary>MNIST</summary>
	<p>Accuracy and loss plots for a ResNet v1 with n = 8.</p>
	<p><img src="outputs/plots/resnet-v1n8-mnist-accuracy.png" width="400">
	<img src="outputs/plots/resnet-v1n8-mnist-loss.png" width="400"></p>
	<p>Accuracy and loss plots for the Mini GoogLeNet variant</p>
	<p><img src="outputs/plots/minigooglenet-mnist-accuracy.png" width="400">
	<img src="outputs/plots/minigooglenet-mnist-loss.png" width="400"></p>
	<p>Learning curves of the VGG-16 variant model</p>
	<p><img src="outputs/plots/vgg16-MNIST-accuracy.png" width="400">
	<img src="outputs/plots/vgg16-MNIST-loss.png" width="400">></p>
</details>

## Weights
The weights for the trained models can be found under the link:

https://drive.google.com/open?id=16AeENomjhIT1C3vmw1fc85xP2SSlZe5z

## References
<a id="1">[1]</a>
Karen Simonyan and Andrew Zisserman, (2014).
Very Deep Convolutional Networks for Large-Scale Image Recognition.
https://arxiv.org/abs/1409.1556

<a id="2">[2]</a>
Christian Szegedy and Wei Liu and Yangqing Jia and Pierre Sermanet and Scott Reed and Dragomir Anguelov and Dumitru Erhan and Vincent Vanhoucke and Andrew Rabinovich, (2014).
Going Deeper with Convolutions.
https://arxiv.org/abs/1409.4842

<a id="3">[3]</a>
Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, (2015).
Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385

<a id="4">[4]</a>
Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, (2016)
Identity Mappings in Deep Residual Networks.
https://arxiv.org/abs/1603.05027