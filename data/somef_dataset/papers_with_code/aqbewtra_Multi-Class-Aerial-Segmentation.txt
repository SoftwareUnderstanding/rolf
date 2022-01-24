# Multi-Class-Aerial-Segmentation

This model is an implementation of UNet for semantic segmentation in the PyTorch framework. The implementation is aimed to be modular, in that it can be adapted for use with other datasets, models, implementations, etc.

The goal of this model is to identify the pixel boundaries of geographic features. In development, I hope to:
* Develop a resilient segmentation model that can adapt to real world implementation.
* Implement a pipline to find a useful form of geographic feature identification.
* Set a helpful basis for further development in data augmentation, and model development for better results.
  

## Resources
* UNet - https://arxiv.org/abs/1505.04597
* UNet++ - https://arxiv.org/pdf/1807.10165.pdf
* Dataset - https://github.com/dronedeploy/dd-ml-segmentation-benchmark
