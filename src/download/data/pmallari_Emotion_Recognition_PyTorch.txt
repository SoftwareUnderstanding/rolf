# Emotion Recognition

This is an attempt to detect emotion based on facial features. The model determines whether the person is Happy, Angry, Sad, Disgusted, Afraid, Surprised or Neutral. The model has three convolutional layers connected to three fully connected layers. The model is trained on a dataset of 4172 images. Currently, the model is deployed via screen grab which detects a portion of the screen and takes it as input.

![](demo.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Make sure you have the following installed (or the latest version):

```
Python 3.6
Numpy 1.15.4
Pandas 0.23.4
OpenCV2 3.4.4
Torch 1.0.0
TorchVision 0.2.1
```

### Installing

Simply fork notebook into your local directory.

## Deployment

Assuming you have all necessary modules installed. Through your command prompt, move to the local repository and run the command:

```
python GrabScreen.py
```
A window would open mirroring a portion of your screen. Simply move the image over that portion of the screen and the predicted emotion is shown on the upper left.
For best performance, let the face occupy the entire window.

## Authors

* **Prince Mallari** - (https://github.com/pmallari)

## Acknowledgments

* Prudhvi Raj Dachapally (https://arxiv.org/ftp/arxiv/papers/1706/1706.01509.pdf)

* Ian J. Goodfellow (https://arxiv.org/pdf/1302.4389v3.pdf) & (https://www.kaggle.com/c/facial-keypoints-detector)
* David Warde-Farley
* Mehdi Mirza
* Aaron Courville
* Yoshua Bengio

* Harrison Kinsley (https://www.youtube.com/user/sentdex)
