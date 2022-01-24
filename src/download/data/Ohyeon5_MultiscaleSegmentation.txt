# Multi-scale feature map induced image segmentation
Deep neural network is mimicking hierachical and feedforward process of human visual cortex. However, it is not a whole story. Human visual system is rather dynamic and recurrsive, therefore, interactive through out different layers.
Such a top-down and bottom-up interactions are seemed to mimicked as a form of residual layers (or short and long skip connections). However, it is unclear how it is explained with regard to human visual processing. 
In current project, characteristics of mutiple scale residual maps are studied, and their integration strategies are studied. Corresponding features and integration strategies are considered with respect to human perceptual features. 

This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/) and supported by tensorflow Korea, Google, Kakao-brain, Netmarble, SKT, Element AI, JDC, and Jeju Univ.

Fully proposed by **Oh-hyeon Choung** *(PhD candidate, EPFL Neuroscence program)*

Main references:
> 1. Lauffs, M. M., Choung, O. H., Öğmen, H., & Herzog, M. H. (2018). Unconscious retinotopic motion processing affects non-retinotopic motion perception. Consciousness and cognition. [(link)](https://www.sciencedirect.com/science/article/pii/S1053810017305421?via%3Dihub)
>2. Shelhamer, E., Long, J., & Darrell, T. (2016). Fully Convolutional Networks for Semantic Segmentation. ArXiv:1605.06211 [Cs]. [(link)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
>3. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. [(link)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)


## Task: Image semantic segmentation 

examples) 

![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example1.png)
![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example2.png)


**Hypothesis**
1. Feature maps from each convolutional layer include distinct information
2. Depending on it's local/abstract features, could they be integrated using different strategy as human does? 


## Introduction
Human visual system starts from lower visual area and proceed to the higher areas. However, it is not a full story. Our lower visual areas are largely affected by various higher visual area interactively. 

![Retino and Non-retino images][incongOccluded]


## Obejective
1. To 



## Base line model: FCN (fully convolutional network) 
Base line model is forked from https://github.com/warmspringwinds/tf-image-segmentation.git 

For the baseline setting, please refer to original github repository.

### Major Debugging Problems 
- The code is written in python 2 (python 2.7 and tensorflow ==1.9.0 worked for me)
- In python 3 (and python 2 of tf 1.x.x): tf.pack --> tf.stack
- Beaware of tfrecord's file path and name: causes 
- "std::bad_alloc" error: RAM memory out or in border
- ['label' out of range] error: 255 (border) values in label file cuses error. For me I've added 
```
# Take away the masked out values from evaluation
weights = tf.to_float( tf.not_equal(annotation_batch_tensor, 255) )
# Get rid of 255s from the annotation_batch_tensor -> by multiplying weight factor
annotation_batch_tensor = tf.multiply(annotation_batch_tensor, tf.cast(weights,tf.uint8))
```


[incongOccluded]: https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/TPD_blackDisk_cong-incong_occlude.gif
