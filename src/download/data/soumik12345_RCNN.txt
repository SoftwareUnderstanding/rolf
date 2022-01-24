# RCNN (Ongoing)

Tensorflow implementation of the RCNN object detection system as proposed by [Rich feature hierarchies for accurate object detection and semantic segmentation
](https://arxiv.org/abs/1311.2524).

**Test Notebook:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/soumik12345/RCNN/master?filepath=notebooks%2FRCNN_Notebook.ipynb)

**Documented Notebook:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/soumik12345/RCNN/master?filepath=notebooks%2FRCNN.ipynb)

## Pascal VOC2012

### Number of Objects per Image

![](./assets/plot_0.png)

### Frequency Distribution of Classes

![](./assets/plot_1.png)

### Sample Images + Ground Truth

![](./assets/plot_2.png)

![](./assets/plot_3.png)

![](./assets/plot_4.png)

![](./assets/plot_5.png)

![](./assets/plot_6.png)

## RCNN

### Overview

The RCNN system was proposed by **Ross Girshick**, **Jeff Donahue**, **Trevor Darrell** and **Jitendra Malik from UC Berkeley in their paper [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://arxiv.org/abs/1311.2524). RCNN stands for Regions with CNN features, which summarizes the working of the system in very simple terms, generating region proposals with classification using CNNs. The RCNN consists of 3 simple stages:

1. Given an input image, around 2000 bottom-up region proposals are extracted.

2. Computation of features for each proposal using a large convolutional neural network (like pre-trained VGG or ResNets).

3. Classification of each region using class-specific linear SVMs (or MLPs).

![](./assets/img_1.png)**

### Seletive Search

For generating the region proposals, we would look towards the following 2 papers:

1. [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf) proposed by by Pedro F. Felzenszwalb and Daniel P. Huttenlocher.

2. [Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) proposed by J R R Uijlings, K E A van de Sande, T Gevers and A W M Smeulders.

#### Felzenszwalb Segmentation

![](./assets/plot_7.png)

![](./assets/plot_8.png)

![](./assets/plot_9.png)

![](./assets/plot_10.png)

#### Texture Gradients

![](./assets/plot_11.png)

![](./assets/plot_12.png)

![](./assets/plot_13.png)

![](./assets/plot_14.png)

#### Generating Bounding Box Proposals

![](./assets/plot_15.png)

![](./assets/plot_16.png)

![](./assets/plot_17.png)

![](./assets/plot_18.png)

#### Narrowing down on objects using IoU

![](./assets/plot_19.png)

![](./assets/plot_20.png)

![](./assets/plot_21.png)

![](./assets/plot_22.png)

#### Narrowing down on objects using IoU (OpenCV Implementation)

![](./assets/plot_23.png)

![](./assets/plot_24.png)

![](./assets/plot_25.png)


## Citation

```
@misc{1311.2524,
    Author = {Ross Girshick and Jeff Donahue and Trevor Darrell and Jitendra Malik},
    Title = {Rich feature hierarchies for accurate object detection and semantic segmentation},
    Year = {2013},
    Eprint = {arXiv:1311.2524},
}
```