# DL_basic_github

[TOC]

------

## About

This is  a repo to implement basic network model by PyTorch



## DL Model

### 1. LeNet-5

- Paper : http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

- [x] Read the Paper
- [x] Construct the model

### 2. AlexNet

- Paper : https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

- [ ] Read the Paper
- [ ] Construct the model

### 3. GoogLeNet

- Paper : https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
- [ ] Read the Paper
- [ ] Construct the model

### 4. VGGNet

- Paper : https://arxiv.org/abs/1409.1556
- [ ] Read the Paper
- [ ] Construct the model

### 5. ResNet

- Paper : https://arxiv.org/abs/1512.03385
- [ ] Read the Paper
- [ ] Construct the model



## Dependencies

- [Python 2.7 ](https://www.continuum.io/downloads)
- [PyTorch 0.1.12](http://pytorch.org/)

 

## Getting Started

### Create Conda environments

```shell
conda create -n py27 python=2.7
source activate py27
conda install pytorch=0.1.12 -c soumith
conda install torchvision
```

### Run Experiment

```shell
git clone https://github.com/mr-bulb/DL_basic_github.git
cd DL_basic_github/LeNet-5
chmod +x ./run.sh
./run.sh
```

### Visualize The Result

```shell
python visual-error.py
python visual-loss.py
```



## Author

Hao Zeng/ [@mr-bulb](https://github.com/mr-bulb)



## License

Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.