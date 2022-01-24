[//]: # (Image References)

[image1]: static/sample_dog_output.png "Sample Output"
[image2]: static/vgg16_model.png "VGG-16 Model Layers"
[image3]: static/vgg16_model_draw.png "VGG16 Model Figure"

## Project Overview

This repo hosts my capstone exercise to the Convolutional Neural Networks (CNN) 
project in the AI Nanodegree of Udacity! 

In this project, I had to build a pipeline that can be used within 
a web or mobile app to process real-world, user-supplied images.  
Given an image of a dog, my algorithm should be able to identify 
an estimate of the canine’s breed. 

If supplied an image of a human, it will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!


## Prerequisites
0. Install Python 3.6
1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep_dog_breeds
	```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Create and activate an environment: 
	```	
		conda env create -f environment.yml 
		conda activate deep_dog_breeds-env
	```
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

## Log Book

Let's describe this journey.  

I learned that the practice is far ahead of the theory in deep learning :)   


First, I tried to build a CNN from scratch. This model had to learn how to see by seeing only *a few* dog images. 
I experimented with many (many!!) different architectures... 
My main trouble was **OVERFITTING**. Indeed, with only 64 sample images from 133 classes, my from scratch model had difficulties to generalize. 
I took me hours of litterature reading, to handle overfitting, nights of training to test hyperparameters, almost drove me crazy not to do better than 35%! I conclueded, that **one cannot properly learn how to see if it only sees dogs**!

Then, using transfer learning, I could take advantage of a model already *sees*, ie. recognizes patterns, textures, colors... and I *only* neeeded to teach him how to recognize dog breeds. 
With the very same sets preparation, it took me about half an hour to choose and implement the model, trained it for only 5 epochs and get >80% accuaracy. 


## Standing on the shoulders of giants
There are a lot of very inspiring works already done, that I could read and take inspirations from, to quote a few: 
**Medium paper tackling the very same exercise**
* https://towardsdatascience.com/dog-breed-classification-hands-on-approach-b5e4f88c333e
* https://medium.com/@imBharatMishra/dog-breed-classification-with-keras-848b9b1525c1
* https://medium.com/@iliazaitsev/dogs-breeds-classification-with-keras-b1fd0ab5e49c
* https://towardsdatascience.com/dog-breed-prediction-using-cnns-and-transfer-learning-22d8ed0b16c5

**Very instructive course on fast AI**
* https://course.fast.ai/videos/?lesson=6

**On optimization: gradient descent optimizer and batch normalization**
* https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
* https://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose
* https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

**One Shot Learning with Siamese Networks**
I did not really work for me as good as I expected but I took time to do a 'POC' using Siamese network to teach a network to estimate the dissimilarity between two dog pictures. I certainly neeed to refine my architecture, but I took so long to train and test that I did not pursued in that direction... I'll put the exploratory notebook in this repository though. 
* https://www.google.com/search?q=siamese+network&oq=Siamesee&aqs=chrome.1.69i57j0l5.2017j0j4&sourceid=chrome&ie=UTF-8

## Take away
If I were to build a model from scratch, I'll certainly be carefull on the followings, that helped me fight against overfitting: 
- **Dropout:** remove activations at random during training in order to regularize the model
- **Data augmentation:** modify model inputs during training in order to effectively increase data size
- **Batch normalization** : adjust the parameterization of a model in order to make the loss surface smoother
- **Residual**: As explained in fast.ai course, this trick has been discovered by chance and improved my performances, as it did it the imagenet dataset (https://arxiv.org/pdf/1512.03385.pdf)
- **Optimizer & Batch size**: I first added a momentum on my SGD optimizer, to accelerate the training, but I had trouble to adjust together (batch size, learning rate, momentum). I then readed about Adam Optimizer that is adaptative, which indeed was easyer to use. Also, I started with batch size of 64, but the variability on losses across epochs was large, as if the model learned too much before validation check, so I decreased batch size to 20. 
Something that would worse testing that would be instead of Adam, try Optimgrad or SGD with nesterov momentum (need to choose parameters). 

## The Road Ahead

We break the notebook into separate steps. 

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm
* [Step 7](#step7): Compare scratch and transfer results

---
## Let's get our feet wet!
---
	
```
    jupyter notebook dog_app.ipynb
```
