# Python to C++ Project on Separable-CNN in MobileNets Paper.
___
## Introduction
we propose what point affect to accuracy from between Python Package and C raw programming without Package.

## Requirements

- Python code about Separable-CNN
- C code about Separable-CNN
- visual studio 2015
- python 2.7

## Datasets

### Train Image Dataset
- download dataset from this [link](https://www.cs.toronto.edu/~kriz/cifar.html) and put it in this project

### Test Image Dataset
The test image dataset are sampled from this [link](https://www.cs.toronto.edu/~kriz/cifar.html) and put ti in this project

## TODO
* Back propagation comparing between Python Package and C without Package just raw programming

## DONE
* [Paper](https://arxiv.org/abs/1704.04861)
* Python Package programming study
* forward propagation C coding(CNN, LightNormalization, Relu, Pool, Depthwise-CNN, Pointwise-CNN, Affine, Softmax)
* Comparing layer by layer
* inference 1 dataset (cifar10)
* valid dataset
* Get Input Image about Normal airplane


> Image Input

 ** Input size Reduction **

- 788 * 526 size image source
<table>
  <tr>
    <td>
     <img src="image/airplane788_526.JPG"/>
    </td>
  </tr>
</table>

- 32 * 32 size input
<table>
  <tr>
    <td>
      <img src="image/airplane32_32.jpg"/>
    </td>
  </tr>
</table>
  



## Reference
- Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias
Weyand, Marco Andreetto, Hartwig Adam. MobileNets: Efficient Convolutional Neural
Networks for Mobile Vision Applications. arXiv:1704.04861, 2017
