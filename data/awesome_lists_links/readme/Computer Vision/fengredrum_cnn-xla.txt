# CNN-XLA

![License Badge](https://img.shields.io/badge/python-3.5%2B-blue) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-TPU.ipynb)

A collection of CNN models are trained on Cloud TPU by using PyTorch/XLA. The performance of these models are only tested on the CIFAR-10 dataset due to the limited computational resources, but it is easy to modify them to fit in more complex datasets (i.e., ImageNet 2012 classification dataset).

# Get Started

* [Train on GPU](notebooks/Train-on-GPU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-GPU.ipynb)
* [Train on TPU](notebooks/Train-on-TPU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-TPU.ipynb)

# CNN Models

| Model              | Input Resolution   | Params(M)          | MACs(G)            | Percentage Correct |
| ------------------ | :----------------: | :----------------: | :----------------: | :----------------: |
| [AlexNet](models/alexnet.py)   | 32x32 | 46.76 | 0.91 | 84.9% |
| [VGG-11](models/vgg.py)        | 32x32 | 28.14 | 0.17 | 69.2% |
| Inception                      | 32x32 | -     | -    | - |
| [ResNet-18](models/resnet.py)  | 32x32 | 11.17 | 0.56 | 88.3% |
| [DenseNet-121 (k = 12)](models/densenet.py)  | 32x32 | 1.0   | 0.13 | 90.5% |
| [SE-ResNet-50 (r = 16)](models/se_resnet.py) | 32x32 | 26.05 | 1.31 | 91.4% |
| [MobileNet-V1](models/mobilenet_v1.py)       | 32x32 | 3.22  | 0.05 | 85.1% |
| [MobileNet-V2](models/mobilenet_v2.py)       | 32x32 | 2.3  | 0.1 | 88.5% |

All of the above models are trained for just 20 epochs with a mini-batch size of 256, learning rate of 0.001 and standard data augmentation. Moreover, the [Mish activation function](https://arxiv.org/abs/1908.08681) is used for better performance.

The goal of this repository is to implement the core concept of a variety of CNN models, so no fancy tricks are used.

# Related Repositories

* [Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
* [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* [PyTorch/XLA](https://github.com/pytorch/xla)

# License

[MIT License](LICENSE)
