# RetinaNet for Object Detection

[RetinaNet](https://arxiv.org/abs/1708.02002) is an efficient one-stage object detector trained with the focal loss. This repository is a TensorFlow2 implementation of RetinaNet and its applications, aiming for creating a tool in object detection task that can be easily extended to other datasets or used in building projects. It includes

1. source code of RetinaNet and its configuration (multiple GPUs training and detecting);
2. source code of data (RetinaNet's inputs) generator using multiple CPU cores; 
3. source code of utilities such as image/mask preprocessing, augmetation, average precision (AP) metric, visualization and so on;
4. jupyter notebook demonstration using RetinaNet in training and real-time detection on some datasets. 


### Updates
* soon/2022: Will have an update to clean up some mess and provide a tutorial on how to generate a customized dataset and then train.
* 10/2/2021: Solve OOM problem when inferencing by fixing resnet_fpn.compute_fmap().

### Applications

The following are example detections.

* [The Global Wheat Challenge 2021](https://www.aicrowd.com/challenges/global-wheat-challenge-2021) is a detection and counting challenge of wheat head. By using this implementation and trained only on the given training set, we are able to achieve the following result (evaluated on the test set used for competition submission):

|GPU| size| detection time (second per image)| evaluation metric (ADA)|
|---|---|---|---|
|GeForce RTX 2070 SUPER|1024x1024|0.11|0.478|

where the evaluation metric ADA is Average Domain Accuracy defined in [here](https://www.aicrowd.com/challenges/global-wheat-challenge-2021#evaluation-criteria). 
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/wheat_movie.gif" width='360' height='360'/>
</p>


* Video detection in human faces: 

https://user-images.githubusercontent.com/38026940/132159211-6951ba51-9d59-4d38-b13e-259504195ebc.mp4

Scenes are taken from *The Bourne Ultimatum (2007 film)* and the cover page is from *The Bourne Identity (2002 film)*. It was trained on the [wider face](http://shuoyang1213.me/WIDERFACE/) dataset. 

Moveover, it can be used to recognize Jason Bourne. See the next video and [ProtoNet for Few-Shot Learning in TensorFlow2 and Applications](https://github.com/DrMMZ/ProtoNet) for details.

https://user-images.githubusercontent.com/38026940/132160401-ee1f22ca-0b0f-4471-8b62-6144c76cf21c.mp4


* My own dataset, *empty returns operations (ERO-CA)*, is a collection of images such that each contains empty beer, wine and liquor cans or bottles in densely packed scenes that can be returned for refunds in Canada. The goal is to count the number of returns fast and accurately, instead of manually checking by human (specially for some people like me who is bad on counting). The dataset (as of July 15 2021) consists of 47 labeled cellphone images in cans, variety of positions. If you are interested in contributing to this dataset or project, please [email](mailto:mmzhangist@gmail.com) me.
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/ero_movie.gif" width='360' height='360'/>
</p> 


* The [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) dataset, focusing on detection in densely packed scenes. Indeed, our ERO-CA detection above used transfer learning from SKU-110K.
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/sku_movie.gif" width='360' height='360'/>
</p>


* The [nuclei](https://www.kaggle.com/c/data-science-bowl-2018) dataset, identifying the cells’ nuclei. 
<p align="center">
  <img src="https://raw.githubusercontent.com/DrMMZ/drmmz.github.io/master/images/nuclei_movie.gif" width='360' height='360'/>
</p> 


### Requirements
`python 3.7.9`, `tensorflow 2.3.1`, `matplotlib 3.3.4`, `numpy 1.19.2`, `opencv 4.5.1`, `scipy 1.6.0`, `scikit-image 0.17.2` and `tensorflow-addons 0.13.0`

### References
1. Lin et al., *Focal Loss for Dense Object Detection*, https://arxiv.org/abs/1708.02002, 2018
2. *Mask R-CNN for Object Detection and Segmentation*, https://github.com/matterport/Mask_RCNN, 2018
