# ESPCN-SuperResolution-Tensorflow
Using ESPCN Model for Improving Dimensions of small Objects(LPR)
## Discription
> A PyTorch implementation of ESPCN based on CVPR 2016 paper [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158.pdf).The proposed efficient sub-pixel convolutional neural network (ESPCN), with two convolution layers for feature maps extraction, and a sub-pixel convolution layer that aggregates the feature maps from LR space and builds the SR image in a single step.the architecture of model is proposed in below:
> 
![Architecture](https://user-images.githubusercontent.com/53394692/111528646-a8a1d180-8776-11eb-81d3-9abf36b10389.PNG)
## Structure of this Project
> * for training,you must use `ESPCN.ipynb` and run it cell by cell.
> * you must construct the `input` directory and you put your dataset in it.
> * the `weights` of trained neural network is constructed in `export` directory.
>
> HINT : i use these weights in my project for license-plate recognition,so if you would like to apply these weights in real problem (with opencv) ,please see my github repo about [opencv-SuperResolution](https://github.com/behnoudshafizadeh/opencv-SuperResolution).
## Training Procedure
> use following commands,step by step basis on `ESPCN.ipynb`:
>  * 1. mount your google drive basis on your path.
```
from google.colab import drive
drive.mount('/content/drive/')
import os
os.chdir("/content/drive/MyDrive/TF-ESPCN-master")
!ls
```
>  * 2. only using tensorflow version 1.x .
```
%tensorflow_version 1.x
import tensorflow
tensorflow.__version__
```
>  * 3. train
```
!python main.py --train --scale 4 --traindir /content/drive/MyDrive/TF-ESPCN-master/input
--scale : you can use 2,3 or 4 for getting 2x,3x,4x output image
--traindir : you can set your image path
```
>  * 4. export weights of trained model
```
!python3 main.py --export --scale 4
```
> after,ending these processes,your weights will ready and you can use these weights in your task.


