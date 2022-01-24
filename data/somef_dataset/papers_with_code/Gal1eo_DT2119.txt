# This is a Music Generation Implementation using WaveNet

This repository contains unconditinal WaveNet structure from:

-WaveNet: A Generative Model for Raw Audio[https://arxiv.org/abs/1609.03499]

The dataset is from:

-The MAESTRO Dataset V1.0.0[https://magenta.tensorflow.org/datasets/maestro#dataset] which stands for MIDI and Audio Edited for Synchronous TRacks and Organization.

## Content

| Section | Description |
|-|-|
| [Theory](#Theory) | Basic Theory |
| [Requirements](#Requirements) | How to install the required package |
| [Usage](#Usage) | Quickstart examples |
| [GPU](#GPU) | GPU requirement and memory |


## Theory

### The Dilated Convolution

The dilated convolution is seen as below:
![Dilated Convolution](https://drive.google.com/uc?export=view&id=1-hDX23TzHGA270vqnSuBVbEMjPbqQfAE)

### The Music WaveNet

The network architecture is seen below:
![Music WaveNet Train](https://drive.google.com/uc?export=view&id=1QKjEvd4HHBTQVHgORhLhcYho4hGvBDau)

![Music WaveNet Generate](https://drive.google.com/uc?export=view&id=1AOXf4xBtR8yWU5rxJ0kxA9RNC6HeAE5q)

### Reduced Version
In order to decrease the complexity of computation, we change the settings as follows:

Total layers: 8
residual channels: 32
skip channels: 128
max dilation: 128

## Requirements

This repo was tested on Python 3.7.3 with PyTorch 1.1 and Scipy 1.3.0

### Installation

PyTorch can be installed by conda as follows:
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
Scipy can be installed by conda as follows:
```bash
conda install -c anaconda scipy
```

## Usage

If you want to reproduce the results music reconstruction, you can run the command:
```bash
python train.py
```
If you want to train on different dataset, you should change the config.json file and train_files.txt

## GPU

If you want to reproduce our results with the defult settings, you need a GPU with more than 10GB memory. Otherwise you need to decrease the number of layers.





