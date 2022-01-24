# Race Events Recognition
Predicting meaningful events in the car racing footage using three-path approach. 

## Table of Contents

* [Project](#race-events-recognition)  
* [Table of Contents](#table-of-contents)  
* [Overview](#overview)  
  * [Description](#description)  
  * [Inspiration](#inspiration)
  * [Solution Overview](#solution-overview)  
* [Prerequisites](#prerequisites)
  * [Tools](#tools)
  * [Libraries](#libraries)
* [Experiments](#experiments)
  * [Dense Optical Flow](#dense-optical-flow)
  * [Sparse Optical Flow](#sparse-optical-flow)
  * [Focused Optical Flow](#focused-optical-flow)
  * [Embeddings](#embeddings)
  * [Self-Similarity Matrix](#self-similarity-matrix)
  * [Camera Views Classification](#camera-views-classification)
* [Usage](#usage)
* [Credits](#credits)

## Overview   

### Description
This project is dedicated to the investigation of methods for predicting
meaningful events in footage of car racing. This repository is focused on the
exploration of **collision detection** but contains a tool for the classification of  as well. During the work on this project we've also developed a
**monadic pipeline** library [mPyPl](https://github.com/shwars/mPyPl) to
simplify tasks of data processing and creating complex data pipelines.   
Due to the small amount of data in this problem; we could not rely on neural
networks to learn representations as part of the training process. Instead; we
needed to design bespoke features, crafted with domain knowledge. After series
of experiments, we've created a model based on features obtained using three
different approaches: 

* Dense Optical Flow
* VGG16 embeddings 
* A special kind of Optical Flow - [Focused Optical Flow](#focused-optical-flow).  


ℹ️ *After the release of mPyPl as a independent [pip-installable framework](https://pypi.org/project/mPyPl/), some experimental notebooks in the ```notebooks``` folder have not been updated, but may contain interesting things to explore.*

### Inspiration

* [Two-stream ConvNets for Action Recognition in Videos​](http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf) by Keren Simonyan, Andrew Zisserman
* [Review of Action Recognition and Detection Methods​](https://arxiv.org/abs/1610.06906.pdf) by Soo Min Kang, Richard P. Wildes
* [Focal Loss for Dense Object Detection​](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, et. al.
* [Using Optical Flow for Stabilizing Image Sequences](http://www.cs.utoronto.ca/~donovan/stabilization/stabilization.pdf) by Peter O’Donovan

### Solution overview
Our solution consists of the three main paths (see illustration below). A video
is fed into the three algorithms in parallel mode. Each of them will be
described in details in the [Experiments](#experiments) section. Each output of
the three paths is processed by a separate neural network and then the results
are combined for the final prediction.  
<p align="center">
<img src="content/ml-workflow.png" height="450">
</p>

This project was preceded by another one dedicated to the detection of racing cars using RetinaNet. RetinaNet was trained on [Azure Batch AI](https://azure.microsoft.com/en-us/services/batch-ai/) using [Horovod](https://github.com/uber/horovod) for distributed training. In the near future, we plan to implement the model inference using [Azure Batch](https://docs.microsoft.com/en-us/azure/batch/) to reduce the time spent on video processing.
<p align="center">
  <img src="content/ml-cloud.png" height="250">
</p>

## Prerequisites

### Tools

* [Azure DSVM](https://azure.microsoft.com/en-gb/services/virtual-machines/data-science-virtual-machines/)
* [Jupyter Notebooks](http://jupyter.org/)

### Libraries
In addition to the standard set of data science packages, we've used the following:
* [Keras on Tensorflow](https://keras.io/)
* [opencv-python](https://github.com/skvark/opencv-python)
* [mPyPl](https://github.com/shwars/mPyPl)
* [keras-retinanet](https://github.com/fizyr/keras-retinanet)

To successfully run the **collision recognition** examples, you need to install all the requirements using 
```bash
pip install -r requirements.txt
``` 
 and clone all the content of [keras-retinanet](https://github.com/fizyr/keras-retinanet) repository to ```research/retina``` folder.  
 ```bash
git clone https://github.com/fizyr/keras-retinanet.git
mv keras-retinanet/keras_retinanet/ race-events-recognition/research/retina/ 
```  
To run the **scene detection** example, you need to have installed:
* [Visual Studio 2017 Version 15.7.4 or Newer](https://developer.microsoft.com/en-us/windows/downloads)
* [Windows 10 - Build 17738 or higher](https://www.microsoft.com/en-us/software-download/windowsinsiderpreviewiso)
* [Windows SDK - Build 17738 or higher](https://www.microsoft.com/en-us/software-download/windowsinsiderpreviewSDK)
* [Win2D](https://github.com/Microsoft/Win2D)

## Experiments

### Dense Optical Flow

Our Dense Optical Flow approach originated from [this](https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html) tutorial. It is a complete vector field showing movement of every pixel between frames​. Such features can show not only changes in the movement of the car, but also the style of the camera operator, which may be different during a normal race and an accident. 
<p align="center">
  <img src="content/dense1.jpg" height="82">
</p> 
<p align="center">
  <img src="content/dense2.jpg" height="82">
</p>  

We've also applied a technique of video stabilization described by Peter O’Donovan in his [article](http://www.cs.utoronto.ca/~donovan/stabilization/stabilization.pdf). 
<p align="center">
  <img src="content/dense3.jpg" height="82">
</p>  

In general, the process consists of the following steps: 
<p align="center">
  <img src="content/denseflow.png" height="110">
</p>

### Sparse Optical Flow

The strategy of Sparse Optical Flow is based on [the same](https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html) example. The main idea was to use the changes in the trajectory of a car as input features for the model. Our experiments showed that the flow for normal situation is different from the flow when an accident occurs. 
<p align="center">
  <img src="content/optical1.jpg" height="100">
</p>
<p align="center">
  <img src="content/optical2.jpg" height="100">
</p>
Ok, looks good, right?    
On the other hand, sometimes the standard Optical Flow approach tracks anything but the cars. The Optical Flow algorithm starts with detection of good features to track (regardless of the semantic of the frame) - just edge detection. Therefore, sometimes we do not get the flow of a car, but something extraneous. On the picture below, the algorithm is trying to track the scoreboard.  
  
<p align="center">
  <img src="content/optical3.jpg" height="102">
</p>
The improved Focused Optical Flow algorithm helps us to eliminate this drawback.

### Focused Optical Flow
The main idea of the Focused Optical Flow is based on providing the standard Optical Flow algorithm with the correct areas of interest. For this purpose, we use a trained RetinaNet object detector. Focusing on the detected areas with cars, the algorithm can select appropriate points for tracking. 
<p align="center">
  <img src="content/optical4.jpg" height="102">
</p>
So, the whole pipeline for the Focused Optical Flow looks is shown on the figure below and in general, the post-flow steps are similar to the Dense Flow approach:
<p align="center">
  <img src="content/focusedflow.png" height="110">
</p>

### Self-Similarity Matrix

Initially we used a [Self-Similarity
Matrix](https://en.wikipedia.org/wiki/Self-similarity_matrix) of cosines of
normalized VGG16 embeddings frames as a 2d feature for a CNN encoder.
Theoretically; it's telling us about the regionality and structure of the video
as a function of co-activation in the convolved embedding space. We intend to
use this an additional input to the overall model as it will likely capture some
useful temporal information about scene structure. 

![alt text](content/self-sim.jpg "Self similarity matrix")


### Embeddings
In this case we use pretrained [VGG16](https://keras.io/applications/#vgg16) model to extract features for each frame of the video. After extraction, the features are stacked into two-dimensional vectors and fed into the CNN.  

### Camera Views Classification
The application gives you the ability to score images and video files frame by frame based on onnx model. The model was trained using [Custom Vision](https://www.customvision.ai/) service and exported to be used in UWP APP written on C#. The app is based on [example](https://github.com/Microsoft/Windows-Machine-Learning) of Windows ML SDK from and extended to use with video files.  
There are 6 different classes the model works with: 
1. Pitstops 
2. 1st party view 
3. 3d party view 
4. Command center 
5. People
6. 1st party back view

## Usage
* The **training** process is described in the ```combined-training.ipynb``` notebook. 
* To run the **inference** process use the ```video-pipeline.ipynb``` notebook.
* All the working notebooks with our **experiments** are in ```notebooks``` folder (even though some notebooks are outdated, they contain interesting ideas).
* The ```research``` folder contains our main **python modules**.  
* The ```utils``` folder contains different **useful things** (e.g. visualization tools). 



## Credits
Project team:
* [Dmitry Soshnikov](https://github.com/shwars)
* [Yana Valieva](https://github.com/vJenny)
* [Tim Scarfe](https://github.com/ecsplendid)
* [Evgeny Grigorenko](https://github.com/evgri243)
* [Victor Kiselev](https://github.com/Gaploid)