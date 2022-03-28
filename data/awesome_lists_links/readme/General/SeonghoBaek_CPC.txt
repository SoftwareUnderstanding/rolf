# Contrastive Predictive Coding #
## Tensorflow Implementation of CPC v2: Data Efficient Image Recognition with CPC ##

>#### Reference Paper: https://arxiv.org/abs/1905.09272 ####

##### Unsupervised pretraining based on CPC encoding. #####

##### Model Summary #####

>Input: [96 x 96 x 3]
>Patch: [24 x 24 x 3] 
>Overlap size 12. 1 input image is composed with 49 patches.

>Flow: Input Image -> Make Patches -> Encoding -> Pixel CNN -> CPC

>Encoder: ResDense Block + Global AVG Pool, No Pooling Layer(Conv Only), Batch Norm(Fine Tune Only) and Self Attention.

>***ResDense Block is*** dense convolution block with residual connection(See JointCenterLoss). 

***Contact: seonghobaek@gmail.com***
