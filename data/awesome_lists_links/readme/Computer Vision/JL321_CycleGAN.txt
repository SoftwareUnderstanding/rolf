# Cycle Consistent Generative Adversarial Networks

Tensorflow implementation of (https://arxiv.org/pdf/1703.10593.pdf). 

## Overview

Cycle Consistent GANs are an adaptation of Generative Adversarial Networks, in which the resulting model has the capability of performing domain adaptation between two datasets of varying domains. Again- unpaired! Images between datasets don't need to be directly matched, as the additional cycle consistency term added within the CycleGAN model allows for additional stability within training - of which pressures output domains to be consistent.

Sample mappings shown above.

![Mappings](https://camo.githubusercontent.com/2fadde78dccf4d61f1294933c3e8083c07a303c7/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f6f626a656374732e6a7067)

## Prerequisites

Dataset available at https://github.com/junyanz/CycleGAN (horse2zebra preferred, else change image dimensions). Alter path names in main for local directory for proper usage. 

Packages Required in Environment:
- Tensorflow
- CV2
- Numpy
- Matplotlib

GPU training is preferred.

## Execution

To execute the program, use the following command whilst in terminal:
```
python main.py
```
