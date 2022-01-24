# Sign Language Digits - CNN

## Description
This project is about using the [Sign Language Digits Dataset](https://www.kaggle.com/ardamavi/sign-language-digits-dataset/data) to classify images
of sign language digits. This is similar to the MNIST dataset that has been used throughout the years to classify a grayscale, handwritten digits
between 0 to 9.

## Goal
The idea behind this project is to create a convolutional neural network (CNN) model to classify a sign language digit image to a digit between 0 to 9.

Furthermore, this demonstrates how I approach designing of a learning model and its development. I will outline the designs I have considered and evaluating
their performance in this dataset.

## About the dataset
This dataset has been provided by **Turkey Ankara Ayrancı Anadolu High School** and I have found this dataset through Kaggle. The images are converted
to grayscale images of size 64 x 64.

## Technologies used
In this project, I will be using **Python** as the programming language of choice. Also, I will use the **Keras** framework to create the layers of the CNN model.

## How to run
Type in the following command:
```
python run.py
```

Ensure that all requirements (found [here](requirements.txt)) have been met in order to run the project.

## Architecture
The architecture used in this model is the following

- Convolution 1D: 32 filters and 3 x 1 kernel size
- Maximum Pooling 1D: 2 x 1 kernel size
- Convolution 1D: 64 filters and 3 x 1 kernel size
- Maximum Pooling 1D: 2 x 1 kernel size
- Convolution 1D: 128 filters and 3 x 1 kernel size
- Maximum Pooling 1D: 2 x 1 kernel size
- Convolution 1D: 256 filters and 3 x 1 kernel size
- Maximum Pooling 1D: 2 x 1 kernel size
- Flatten
- Dense: 1024 hidden units
- Dropout: 0.5 hidden unit drop probability
- Dense: 512 hidden units
- Dropout: 0.5 hidden unit drop probability
- Dense: 256 hidden units
- Dense: 10 output units corresponding to digits 0 to 9

This architecture is inspired by the VGG16 network with the paper found [here](https://arxiv.org/pdf/1409.1556.pdf). In this paper, **configuration A** has been used as the starting point.

## Process arriving to the final architecture
Due to the small amount of data, I have to ensure that the amount of parameters is kept to a small amount to ensure that it does not overfit the training set. 
As a result, I have limited the number of convolution operations performed in each of the pixels.

Also, the number of hidden units in this architecture is reduced as it gets closer to the output layer. The application of dropout in between each dense layer 
helps to reduce the effect of overfitting.

## Result
The highest test set accuracy received after 50 epochs is **93.46%**.

The training set accuracy is 99.39% and the validation set accuracy is 88.48%.

The loss function for the training and validation sets is shown here:
![](loss.png)

## Credits
The dataset and its original source can be found through Kaggle's website [here](https://www.kaggle.com/ardamavi/sign-language-digits-dataset/data).

The arXiv paper that refers to the VGG16 network can be found [here](https://arxiv.org/pdf/1409.1556.pdf).
