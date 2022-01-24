## Wheat head counting using EfficientDet D6

A neural network implementation for detecting wheat heads from field optical images using EfficientDet, an object detection model developed by Google Research.
The model relies on the [EfficientNet](https://arxiv.org/abs/1905.11946) architecture and has achieved SOTA results in many benchmarks,
including the COCO Dataset.

This implementation uses a Pytorch version of the model and starts with pre-trained weights. The model is trained with the
[Global Wheat Dataset](http://www.global-wheat.com/), the first large-scale dataset for this task. During training, many augmentation
techniques are used, including brand new Cutmix and Mixup (see references).

The training and inference process are shown in the notebooks, which includes sample images.

### What is the purpose of this model and dataset?

To get large and accurate data about wheat fields worldwide, plant scientists use image detection of "wheat heads"—spikes atop the plant containing grain.
These images are used to estimate the density and size of wheat heads in different varieties.
Farmers can use the data to assess health and maturity when making management decisions in their fields.

### References

Global Wheat Head Detection (GWHD) dataset [[arXiv](https://arxiv.org/pdf/2005.02162)]

EfficientDet: Scalable and Efficient Object Detection [[arXiv](https://arxiv.org/pdf/1905.11946)]

Mixup: Beyond Empirical Risk Minimization [[arXiv](https://arxiv.org/pdf/1710.09412)]

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features [[arXiv](https://arxiv.org/pdf/1905.04899)]

EfficientDet Pytorch: A Pytorch implementation by rwightman [[GitHub repo](https://github.com/rwightman/efficientdet-pytorch)]

Albumentations library [[GitHub repo](https://github.com/albumentations-team/albumentations)]

A huge thanks to Alex Shonenkov for his work with the baseline for this implementation. This code was tested using Python 3.8.2, PyTorch 1.4 and Ubuntu 20.04.
