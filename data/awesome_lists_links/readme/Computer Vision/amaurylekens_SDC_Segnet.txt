
# Artificial Intelligence Laboratory - <br/>Self Driving Car Simulation
## Encoder-Decoder for road Segmentation

@ ECAM 2019

This project consists of designing and developing an AI able to drive a car to keep it on the road in a simplified environment. 
This repository contains the files necessary to realize the semantic segmentation of an image to identify the shape of a road.

(*REM*: another solution that is part of the same project has been implemented. It uses a simple [Convolutional Neural Network](https://github.com/amaurylekens/tensorflow-self-driving-car).)

The simulator is part of the *Udacity's Self-Driving Car Simulator* project available at the following address https://github.com/udacity/self-driving-car-sim

### Segmentation

The purpose of the segmentation is to assign the pixels belonging to the route to one class and the other pixels to another class.

![segnet](https://github.com/amaurylekens/SDC_Segnet/blob/master/images/segnet.png)

The algorithm used is an encoder-decoder neural network. The input of the network is an RGB image and the output is an image (same size) where each pixel is assigned to a class. 

The encoder is composed of several layers of convolution, normalization and pooling. The decoder has an inverse architecture and the pooling layers are replaced by upsampling layers.

The goal is to use this architecture in combination with a CNN to predict the direction of the car in the Udacity simulator.

### Labelization

To train the model, one needs a dataset containing images with their labeled version. To achieve this, we use the labelizer.py function, this function uses image processing techniques to label an image. The main steps are:

![segnet](https://github.com/amaurylekens/SDC_Segnet/blob/master/images/labelization.png)

* Perform two edge detections: a sensitive and a less sensitive
* Select the upper part of the sensitive and the lower detection of the less sensitive and paste them
* Find the first white dots on the left and right from the middle line
* Find the best regression for each edge from these points

## Getting started

### Dependencies

* Car simulator from Udacity
* Python libraries : keras, tensorflow, scikit-learn, openCV, pandas, numpy, matplotlib, base64, io, python-socketio, eventlet

### Installing

The model available on this repository has **already been trained** with images registered locally. The model is stored in the .hdf5 file.

```bash
├── segnet.py
├── train.py
├── labelizer.py
├── test 
│   ├── test.py
│   └── result.png
├── prepare_label.py
├── compute_output_img.py
├── live_segmentation.py
├── model_weight.hdf5
├── README.md
└── .gitignore
```

* segnet.py : a class which contains the encoder-decoder structure with *train* and *predict* methods
* train.py : fits the weights of the network and store them
* labelizer.py : function which helps to create the labels from a set of image
* prepare_label.py : functions which prepares data and labels for the training
* compute_output_img.py : function which transforms the output of the network in segmented image
* live_segmentation.py : road segmentation in live with the udacity simulator
* test.py : tests the model on new image
* model_weight.hdf5 : stores the weights of the trained model

## Running

### Train

1. Prepare a folder with images captured with the Udacity simulator
2. Run the train.py file

### Live segmentation

1. Launch the *beta_simulator* application in autonomous mode and select the left track
2. Run the *live_segmentation.py* file
3. Run the *stream.py* file
4. Drive the car by pressing the W key (there is no self-driving mode for the moment)

## Result

Training on 140 segmented images with 25 epochs : 

<p align="center">
  <img src="https://github.com/amaurylekens/SDC_Segnet/blob/master/test/result.png"/>
</p>

## Author 

* **Lekens Amaury**
* **Wéry Benoît**

## References

* https://medium.com/coinmonks/semantic-segmentation-deep-learning-for-autonomous-driving-simulation-part-1-271cd611eed3
* https://arxiv.org/pdf/1511.00561.pdf

## Acknowledgments

* CBF, HIL, LUR
