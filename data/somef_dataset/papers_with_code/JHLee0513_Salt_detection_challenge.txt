# Salt_detection_challenge
This is a documentation summarizing my approach to the image Segmentation project based on Kaggle's TGS salt identification challenge. 

# What is in this documentation:
* Competiton description
* Motivation
* Data
* Final Model
* Squeeze and Excitation - SE_modules.md
* ResNet - ResNet.md (TODO)
* Unet - UNET.md (TODO)
* Transfer Learning - TLearning.md (TODO)
* Final code - .ipynb
* Augmentation & Ensembling method - misc.md (TODO)

# Competition Description
"Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface.
But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings. More alarmingly, it leads to potentially dangerous situations for oil and gas company drillers." - from competition description.

In this competition TGS provided images collected using seismic reflection of the ground in various locations. Thus in the data we are given training data as the images and their appropriate masks highlighting the salt deposit within that image as labels. The goal of the competition is to build a model that best performs this image segmentation task.

# Motivation
I have taken interest in computer vision due to my recent involvement with the robotics club, and this competition was timely there for me to learn and practice image segmentation task. Thus this was a learning experience for me, not necessarily for winning. 

# The data
<img src="https://math.berkeley.edu/~sethian/2006/Applications/Seismic/smooth_elf_post_img.jpg" width="400" height="300">
The main data of this competition is based on seimic data, example shown above. They are much like ultrasound imaging of the ground. Using waves we can generate images of the subsurface like that above for our segmentation problem. 

# Major ideas implemented in this project:
* Image segmentation model using U-net architecture UpSampling layers in Keras

For our segmentation we use the U-net architecture that has rosen to attention. Full details can be found from the research paper: https://arxiv.org/abs/1505.04597 
<img src="https://cdn-images-1.medium.com/max/1600/1*q3vqSaSTgYzpbk1KIBmWsw.png" width="400" height="300">

With the U-net we are able to build a strong segmentation model, and thus this became the basis of the network architecture.

* ResNet50 as encoder model of the segmentation model
ResNet50, or Residual Net, is one of the state-of-the-art model architectures that I have decided to use for this project. The ResNet is founded on the idea of skip connections to reduce the performance impairment caused by vanishing/exploding gradient that occurs within plain deep network. More can be known about ResNet from https://arxiv.org/abs/1512.03385. Furthermore, in order to increase the performance of the model in catching features Squeeze and Excitation modules, or SCSE modules were added to each convolution block. (https://arxiv.org/pdf/1709.01507.pdf)

* Image augmentation using opencv and numpy
Augmentation is a process where the effective dataset used to train the network is increased by synthetically modifying the train dataset. For instance, horizontal flipping of images essentially double the dataset without necessarily changing the nature of data. Augmentation such as padding, random crop and scale, and flipping have been experimented during this project. In the end I have ended up with no padding but image scaling from original size 101x101 to 128x128 and horizontal flipping.

* Test Time Augmentation (TTA) to predict stronger predictions
Test Time Augmentation is a process where test dataset is augmented to generate multiple predictions, and the final prediction takes average of all predictions made. This increases the chance of the model better capturing the labels from test data. With TTA my Leaderboard (LB) score increased.

* Stratified K-Fold ensembling to develop a better-generalizing model
K-Fold ensembling is a technique multiple version of the model is trained using 'different' dataset and cross validation set. A visualization of a 5-fold validation is as follows:
<img src="https://i.stack.imgur.com/1fXzJ.png" width="400" height="300">
Stratified refers to splitting dataset into fold such that each dataset has equal proprotions of different data. In this project this meant each fold had equally distributed images of varying salt coverage over the image e.g. 5 equal portions of iamges with 50% salt coverage,etc. Implementing Stratified K-Fold increased by LB score as well. 

* Experimented with SGD and Adam optimizers, BCE, DICE, lovas-hinge loss functions
A well-suited optimizer alongside the appropriate loss function certainly improved the model's performance significantly in this project. My final approach ended up with Adam optimizer with a loss function combining binary cross entropy and dice loss. I did not have enough time to further experiment with lovasz loss, unfortunately. 

# Performance & final results:
* Final model - ResNet50 encoder with squeeze and excitation modules with U-Net architecture
* Upsampling layers used instead of transpose concatenating Conv layers
* Adam optimizer
* Cosine annealing with model checkpoints
* Binary Cross Entropy (BCE) with DICE loss as loss function
* Submission: 0.807 Public LeaderBoard score


More to be updated!
