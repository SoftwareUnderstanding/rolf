# Real-time-emotion-recognizer
This is a model trained using FER2013 dataset and later used for real-time emotion recognition. Predicted emotions are visualized using emotes.


This model - 62.6% accuracy.

Kaggle Challenge winner  - 71.2% accuracy. 


![Happy](/test_img_pred/happy_pred.png)

![Neutral](/test_img_pred/neutral_pred.png)

![Angry](/test_img_pred/angry_pred.png)

## Getting started
  I used pytorch for building, training and evaluating the neural network. OpenCV captures frames from the camera and finds all faces in the image. Only a face which covers biggest area of the screen is picked, resized and fed into previosly trained model.
  
  The model does good job in predicting 'happy','angry','neutral','suprised' and 'fear' faces. On the other hand, it does not work very well when it comes to 'sad' and 'disgusted' faces.
  
## Model
  This model is a VGG13 deep convolutional neural network. It is described in this paper: https://arxiv.org/abs/1409.1556
  
## Data set

https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Installation

I'm using the package manager [pip](https://pip.pypa.io/en/stable/installing/) to install Pytorch, Pandas and OpenCV.

https://pytorch.org/ for pytorch installation

```bash
pip install pandas
pip install opencv-python
```

## Usage

##### Real-time emotion recognition
  If you want to run real-time emotion recognition using already built model, just run the camera.py file. To get the best results, your face should be in front of the camera, the source of light should not be behind you.
  
##### Training model
  If you want to train model, run the 'train_model' function in model.py file. Note that this functions needs fer2013.csv dataset to proceed.
  
##### Testing model using FER2013 test set
  If you want to test model using FER2013 test set, run the 'test_model' function in model.py file. Note that this functions needs fer2013.csv dataset to proceed.

  
  
