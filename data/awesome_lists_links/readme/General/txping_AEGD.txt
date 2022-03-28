# AEGD
This repository contains code to reproduce the experiments in "AEGD: Adaptive gradient descent with energy".


## Usage
The aegd.py file provides a PyTorch implementation of AEGD,

```python3
optimizer = aegd.AEGD(model.parameters(), lr=0.1)
```
AEGD with decoupled weight decay (AEGDW) can be constructed by setting `aegdw=True`.
```python3
optimizer = aegd.AEGD(model.parameters(), lr=0.1, aegdw=True)
```
Learn more about `decouple weight decay` at [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)


## Examples on CIFAR-10 and CIFAR-100
We test AEGD(W) on the standard CIFAR-10 and CIFAR-100 image classification datasets, comparing with several baseline methods including: SGD with momentum (SGDM), Adam and AdamW. The implementation is highly based on [this repository](https://github.com/Luolc/AdaBound). We also provide a [notebook](./visualization.ipynb) to present our results for this example.

Supported models for CIFAR-10 are ResNet, DenseNet and CifarNet,  for CIFAR-100 are SqueezeNet and GoogleNet. A weight decay of `1e-4` is applied to all the optimizers. The initial set of step size for each optimizer are:

* SGDM: {0.05, 0.1, 0.2, 0.3}
* Adam: {1e-4, 3e-4, 5e-4, 1e-3, 2e-3}
* AdamW: {5e-4, 1e-3, 3e-3, 5e-3}
* AEGD: {0.1, 0.2, 0.3, 0.4}
* AEGDW: {0.6, 0.7, 0.8, 0.9}

We note that the above setting for initial step size is calibrated for training complex deep networks. In general, suitable step sizes for AEGD(W) are slightly larger than those for SGDM. The best initial step size for each method in a certain task are given in respective plots in our paper to ease your reproduction.

Followings are examples to train ResNet-56 on CIFAR-10 using AEGD with a learning rate of 0.3

```bash
python cifar.py --dataset cifar10 --model resnet56 --optim AEGD --lr 0.3
```
and train SqueezeNet on CIFAR-100 using AEGDW with a learning rate of 0.9
```bash
python cifar.py --dataset cifar100 --model squeezenet --optim AEGDW --lr 0.9
```
The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve will be saved in the `curve` folder.


## License
[BSD-3-Clause](./LICENSE)
