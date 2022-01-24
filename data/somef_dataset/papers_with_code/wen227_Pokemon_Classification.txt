# Pokemon_Classification
基于卷积神经网络的宝可梦分类.

Key Words:Convolutional neural network;Image classification;deep learning; Pokémon.

Pokemon Classification based on CNN, which can classify five classes including Bulbasaur、Charmander、Mewtwo、Pikachu、Squirtle. 

dataset : Contains the five classes, each class is its own respective subdirectory.

example : Contains images that will be using to test our CNN.

result : Contains result pictures and model.

model.py : Convolutional Neural Network model.

train.py : Use this script to train Keras CNN, plot the accuracy/loss.

predict_plot.py : Use this script to predict example images and plot result.

predict_test.py : Use this script to predict textset images and print confusion matrix and accurancy.

# Reference
1.https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

2.https://www.kaggle.com/trolukovich/predicting-pokemon-with-cnn-and-keras/notebook
  
  These two webs above present examples of VGG neural network model.
  
3.https://arxiv.org/abs/1409.1556 VGGNet network

4.https://keras.io/zh/applications/

5.https://cloud.tencent.com/developer/article/1038802
