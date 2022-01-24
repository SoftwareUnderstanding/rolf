# Breast_Cancer_Detection_CNN
A CNN model for detecting Breast Cancer from Images. This project was one of my early attempts of applying Deep Learning(Specifically CNNs) to classification problems.



## Dataset
You can download the dataset through this [link](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

The Dataset contains microscopic images of breast tumor tissue. The images were taken using different magnifying factors.
I have used the 40X - set of images.
1000 for training (500 - Benign ---- 500 - Malignant).
The rest of them were used for validation.

## Libraries and Frameworks
This project was completely implemented using Tensorflow! 
It's easy and saves us a lot of time and effort.

## Model

I have built my model on top of the DenseNet-201 architecture. Imagenet pre-trained weights were used. (See the code)
Since we have a small amount of data available to us, I have used data augmentation to generate some more images. 

https://arxiv.org/abs/1608.06993


## Result

![Train-Val Accuracy](Accuracy.png)

Accuracy achieved :- 98%

If you have any issues or doubts, feel free to ask them. I'll do my best to answer them. :)

Happy Learning People ! Keep chasing your dreams ! ⭐️
