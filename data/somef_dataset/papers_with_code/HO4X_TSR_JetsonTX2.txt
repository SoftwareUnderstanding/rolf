# TrafficSignRecognition for JetsonTX2
### Note: this repo is WIP!

This project splits into two sections: 
- one for the training and engine creation of the CNN-Classifier (MobileNetV1)
- the other one is the runtime code for the Nvidia Jetson TX2

## Basic Idea behind this 
![Alt SW Architecture](doc/idea.jpg?raw=true "SW Architecture")

This approach does run really fast on the Jetson as long as the VisionWorks OpenCV is used, as its optimized. 
Also the runtime depends on the input size of the CNN. 
The current CNN model is an mobilenet, which is able to recognize 43 classes of german traffic signs. 

## Current ToDo's
An overview of the current open points. 

- Add Demo
- !Complete scripts etc.!

### CNN Classifier
- implement EfficientNet (https://arxiv.org/abs/1905.11946)
- Add background class
- Add test dataset

### Jetson Runtime Code 
- improve ROI-Filtering 
- refactor inference for bigger batch size in order to detect faster more signs
- check if there is an possible benefit of using the shared memory between CPU/GPU

