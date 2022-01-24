# Overview
This is just a example to train Fashion-MNIST by Residual Network (ResNet) for learning TensorFlow 2.0.

# Description
* This is an assignment of [Deep Learning basic class](https://deeplearning.jp/lectures/dlb2018/) arranged a little. 
* Training Fashion-MNIST by ResNet on Google Colaboratory with TensorFlow 2.0 Alpha.
* Data is augmented by ImageDataGenerator of Keras.

# Attention
Fashion-MNIST-by-ResNet-50.ipynb file can not be opened correctly, I'm not sure about the reason though. However, you can see the code in Google Colaboratory. So please take a look at [How-to-open-ipynb-on-Google-Colaboratory](https://github.com/shoji9x9/How-to-open-ipynb-on-Google-Colaboratory) and open it in Google Colaboratory.

# Dataset
This dataset can be found [here](https://github.com/zalandoresearch/fashion-mnist).

# Model
ResNet-50 from https://arxiv.org/abs/1512.03385.

# Accuracy
* with Adam optimizer  
91.3% after training in 400 epochs.  
![image](https://user-images.githubusercontent.com/40084422/57193895-51782680-6f7b-11e9-83fc-edb86071f7e2.png)

* with SGD + Momentum  
91.4% after training in 400 epochs.  
![image](https://user-images.githubusercontent.com/40084422/57569816-be853380-7435-11e9-902d-08cc6fed0b42.png)

# How to open it on Google Colaboratory
Please take a look at [How-to-open-ipynb-on-Google-Colaboratory](https://github.com/shoji9x9/How-to-open-ipynb-on-Google-Colaboratory).

