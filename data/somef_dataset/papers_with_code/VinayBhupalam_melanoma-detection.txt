# Melanoma Detection

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical](#technical)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [To Do](#to-do)


## Demo
Trained model is deployed using Heroku Platform
Link: https://detection-melanoma.herokuapp.com/

## Overview
This project is aimed to classifiy an image as malignant or benign and thereby detect skin cancer [Melanoma].

## Dataset and preprocessing
ISIC 2019 Dataset for benign and malignant cases available in Kaggle
JPEG format are converted to TFRecord format for faster processing and used tensor flow dataset class to load the data in batches.
TFRecord files created are uploaded back to kaggle and hence into GCS Cloud.
TFRecord files are created for various sizes of images - 224x224 60% center cropped, 384x384 50% center cropped.

## Motivation
Motivation to do this project is to apply state of the art computer vision techinques to real world applications

## Technical 
Tensorflow deep learning frame work is used.

Model trained on Kaggle TPU

EfficientNetB4 pre trained model is used, Please refer the research paper for the architecture of EfficientNet - https://arxiv.org/abs/1905.11946

Images are augmented with respect to zoom,shear,rotate left/right flip up/down during training randomly.

A significant Class imbalance problem was seen as expected in medical images. Number of positive data points were 584 compared to 30,264 Negative data points
Training data is under-sampled i.e on EDA of the metadata it was seen that for the same part of the body, for the same patient multiple images were found. These images were
of different sizes or taken from a different angle, natural augmentation!!!!. But since we do augmentation during training, we can remove these duplicate images [Checked the performance of the model with all the available data as well, there was almost no difference].

Loss functions experimented are Binary crossentropy, weighted binary cross entropy, focal loss  -> weighted binary cross entropy gave better performance.

Cyclic learning rate, Learning rate with decay,Learning rate decay with warmup has been experimented -> Learning rate decay with warmp gave better results.

Batch size of 8,16,32 has been experimented.

Stratified KFold technique is used for cross validation splits.

AUC achieved is 0.895


## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```
## Run
python app.py

## Deployement on Heroku
Free Heroku account has been created and the model has been deployed.



## To Do
Create an ensemble of models using various architectures/image sizes and meta deta

Dockerize the model




![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.tensorflow.org/images/tf_logo_social.png" width=200>](https://tensorflow.org/) [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) 












