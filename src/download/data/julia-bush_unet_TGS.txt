# unet_TGS

<b>About TGS Salt Identification Challenge</b>

Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface. But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings. More alarmingly, it leads to potentially dangerous situations for oil and gas company drillers.

To create the most accurate seismic images and 3D renderings, TGS (the world’s leading geoscience data company) hoped Kaggle’s machine learning community would be able to build an algorithm that automatically and accurately identifies if a subsurface target is salt or not.


<b>About this repository</b>

This repository contains an implementation of U-net, a convolutional neural network, in Python using Keras, which can be trained to address the above challenge. Only horizontal flip is used for data augmentation. The model is compiled using Adam as optimizer and binary cross-entropy as loss function. It runs for a maximum of 100 epochs with EarlyStopping and ReduceLROnPlateau callbacks, achieveing accuracy of 0.93. The 4000 images with corresponding masks are split 80%-20% for model training and validation purposes respectively. The plot of the learning curve as well as a random selection of visualised predictions are included in the "plots" folder. The dataset is not included here but can be downloaded (link below).

TGS.py takes about 8 minutes to run on i5-6600K core with 16GB RAM and NVIDIA GTX 970 with CUDA/cuDNN installed.


TGS Salt Identification Challenge dataset source:
https://www.kaggle.com/c/tgs-salt-identification-challenge/

About U-net:
https://arxiv.org/pdf/1505.04597.pdf

About Keras:
https://keras.io/about/

Code snippets from:
https://www.depends-on-the-definition.com/unet-keras-segmenting-images/

