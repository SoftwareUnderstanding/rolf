# Finger-Counter
A program to count the number of fingers you're holding out.

# Getting Started

Simply run counter.py or tracker-counter.py. 
counter.py is less volatile than tracker-counter.py,
but the latter is able to track the location of your hand.


# Convolutional Neural Network (ConvNet)
counter.py

![convnet test](./images/CNN_test.gif)

This one is pretty simple. There's a green box and whatever 
is in it is fed into a ConvNet. 

Works best against a white wall in a well lit area.


# Single Shot Detector (SSD) + ConvNet
tracker-counter.py

![sdd convnet test](./images/SNN_CNN_test.gif)

This architecture was heavily inspired by MrEliptik's HandPose model (https://github.com/MrEliptik/HandPose). I also used the SSD he trained.

An SSD detects and places a min-area bounding box around your hand. An image is cropped from it and fed into a ConvNet. 

Works in a well lit area. Background doesn't matter a whole lot, just good lighting. 


# Architecture
The ConvNet architecture is pretty straightforward, and was achieved through lots and lots of trail and error. ![conv net](./images/cnn.png)

I found that more layers is better, and anything more than 32 filters per layer lead to overfitting. Too many layers also lead to overfitting. I trained many many models, and this specific architecture was quite resilient to overfitting fortunately. Anyone who wants to improve on this ConvNet, I recommend you play around with strides and kernel sizes since I did not really alter them. 

***

Single Shot Detector (https://arxiv.org/pdf/1512.02325.pdf) is a bit more complicated. ![SSD](./images/ssd.png) It's quite fascinating how the SSD works. Long story short, the model detects 100 boxes, each box having its own probability distribution of all the classes. And every box has a score that something is in it (since most of the bounding boxes will contain nothing). Thus the SSD outputs a very long array. Let's say you trained it on 3 classes. Each box would correspond to an array of size 8: 4 numbers for the box center, length, and height; 3 numbers for the class distribution, and 1 number for the box score. Since the SSD outputs 100 boxes, this would correpsond to the SSD outputting an 800-element array. In my case, there was only one class: hand.  

Those 100 bounding boxes are filterd through a score threshold (I chose .2). The box with the highest score after that is chosen as the output box. This implies that sometimes no box will be chosen as the output box, which is a good thing if no hand is in the image. Andrew Ng has a good video on this Multibox algorithm (https://www.youtube.com/watch?v=gKreZOUi-O0).


# ToDo
- fine-tune ConvNet a bit more
- train SSD on more data
