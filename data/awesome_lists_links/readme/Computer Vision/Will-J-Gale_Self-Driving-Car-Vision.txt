# Self Driving Car Vision
Vision model development for GTA5 self driving car project

![alt text](https://github.com/Will-J-Gale/Self-Driving-Car-Vision/blob/master/Images/CarVision.gif)  
Top left: original mage
Top right: Segmentation (Road - Blue, Vehicles - Green, Everything else - Red)
Bottom left: Lane segmentation (green)
Bottom right: depth estimation (black = closer, white = farther)

# Introduction
This project will be the backbone for the latest updates coming to the GTA5 self driving car project.
(https://github.com/Will-J-Gale/GTA5-Self-Driving-Car)

# Model
![alt text](https://github.com/Will-J-Gale/Self-Driving-Car-Vision/blob/master/Images/CarVisionModel.png)  
The model is based of UNET architecture: https://arxiv.org/abs/1505.04597

It was trained using 4 separate datasets which:  
* https://download.visinf.tu-darmstadt.de/data/from_games/  
* https://phuang17.github.io/DeepMVS/mvs-synth.html 
* https://bdd-data.berkeley.edu/ 
* http://synthia-dataset.net/  

Currently the model takes in an image and predicts object segmentation, lane prediction
and depth estimation however, this model's outputs will eventually all feed into a final 
convolutional section which will predict the steering and throttle commands to drive 
the car.  

The final small model shown on the left of the image above is the speed prediction.
This is specific to the GTA5 project as there is no direct way of getting the current
speed of the car through the image.

# Testing the model
The TestModel.py script provided gives a demonstration on how to use the model.  
Simply provide a path to an image to the __imageFilepath__ variable and hit run,  
this will create a single image with the original image a all 3 predictions to  
appear on screen.

# More examples
https://github.com/Will-J-Gale/Self-Driving-Car-Vision/blob/master/Images/CarVision2.gif  
https://github.com/Will-J-Gale/Self-Driving-Car-Vision/blob/master/Images/CarVision3.gif  
https://github.com/Will-J-Gale/Self-Driving-Car-Vision/blob/master/Images/CarVision4.gif  
