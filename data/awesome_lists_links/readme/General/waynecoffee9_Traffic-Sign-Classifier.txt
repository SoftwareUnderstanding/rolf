# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/trainingQTY.jpg "trainingQTY"
[image2]: ./examples/validQTY.jpg "validQTY"
[image3]: ./examples/testQTY.jpg "testQTY"
[image4]: ./examples/before_augmentation.jpg "Before Augmentation"
[image5]: ./examples/augmentation.jpg "Augmentation"
[image6]: ./examples/augmentQTY.jpg "Total Training Set"
[image7]: ./examples/incep_overall.jpg "Inception v4, 1"
[image8]: ./examples/my_incep.jpg "My Inception v4 Net"
[image9]: ./examples/validation_recall.jpg "Validation Recall"
[image10]: ./examples/validation_precision.jpg "Validation Precision"
[image11]: ./examples/class_16_41.jpg "Misclassification"
[image12]: ./examples/new.jpg "New Images"
[image13]: ./examples/image1_prob.jpg "Image Top 5 Probabilities"
[image14]: ./examples/image2_prob.jpg "Image Top 5 Probabilities"
[image15]: ./examples/image3_prob.jpg "Image Top 5 Probabilities"
[image16]: ./examples/image4_prob.jpg "Image Top 5 Probabilities"
[image17]: ./examples/image5_prob.jpg "Image Top 5 Probabilities"
[image18]: ./examples/layer_vis1.jpg "Feature map 1"
[image19]: ./examples/layer_vis2.jpg "Feature map 2"
[image20]: ./examples/layer_vis3.jpg "Feature map 3"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/waynecoffee9/Traffic-Sign-Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier-inception.ipynb)
If you are unable to view it under github, use https://nbviewer.jupyter.org/ and paste the link to view.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Three bar charts show image counts by class in training, validation, and test sets.  One can see that image counts vary a lot among classes.  This can potentially negatively affect accuracies for classes with fewer images.  Data augmentation will be introduced later in the preprocessing stage.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I normalized all training images to have float32 from 0 to 1.  I found the accuracy increases faster than -1 to 1 during network training.
The matricies below simply show one random image before and after normalization.

Before normalization:

[[51 36 45 ... 80 79 73]

 [47 34 38 ... 64 75 79]
 
 [45 32 38 ... 61 68 71]
 
 ...
 
 [43 38 34 ... 46 42 37]
 
 [44 36 31 ... 36 33 35]
 
 [41 36 38 ... 52 48 50]]
 
After normalization:

[[0.1849315  0.08219178 0.14383562 ... 0.38356164 0.37671232 0.33561644]

 [0.15753424 0.06849315 0.09589041 ... 0.2739726  0.34931508 0.37671232]
 
 [0.14383562 0.05479452 0.09589041 ... 0.25342464 0.30136988 0.3219178 ]
 
 ...
 
 [0.13013698 0.09589041 0.06849315 ... 0.15068494 0.12328767 0.0890411 ]
 
 [0.1369863  0.08219178 0.04794521 ... 0.08219178 0.06164384 0.07534247]
 
 [0.11643836 0.08219178 0.09589041 ... 0.19178082 0.16438356 0.1780822 ]]

 

As mentioned before, data augmentation is applied to even out image quantity difference among classes, and to include variations of same images.

* sharpen or smoothing
* random rotate image
* random stretch/squeeze image 
* random darken partial image
* random move image

Here is an example of a traffic sign image before and after augmentation.  The image is stretched horizontally and partially darkened at the bottom.

![alt text][image4] ![alt text][image5]

When all training images are added up, the quantity shows:

![alt text][image6]

As a last step, the training set is shuffled to remove any order.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model uses scaled down version of inception v4.

A typical inception v4 net consists of the following architecture (Reference: https://arxiv.org/pdf/1602.07261.pdf):

![alt text][image7]

Each block contains layers of convolutions and pooling in series and parallel.  Please refer to pages 3 and 4 in the reference PDF provided above for the detailed layers.

My inception model has fewer filter depths for faster training time.  See below:

![alt text][image8]

The only fully connected weights are between dropout and the final output layer.  The rest are convolutions and pooling.



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer because it seems to be faster than traditional gradient descent.  There are also other benefits mentioned online (Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

The batch size is 128, which is a typical value.  For number of epochs, I set it to 6.  Every time 6 epochs are done, the trained weights are saved.  I regenerate the whole data augmentation and continue until the accuracies have reached peaks.

For learning rate, I setup maximum rate (also default) as 0.002.  As training set accuracy is closer to 100%, learning rate will be adjusted automatically after each epoch.

```javascript
    learn_rate_fac = 0.02
    default_learn_rate = 0.002
    dyn_alpha = min((1 - train_accuracy)*learn_rate_fac, default_learn_rate)
```

For L2 regularization, beta is set to a fixed value of 0.001.

For dropout, I keep 80% of the weights during training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%+
* validation set accuracy of 99%+
* test set accuracy of 98%+

If a well known architecture was chosen:
* Inception v4 was chosen for traffic sign classifier.
* This is a very suitable architecture because it has a very high accuracy for classifier (general inception v4 can be used to classify 1000 classes), and it is quite efficient.
* It can be concluded this model works very well because all 3 data sets have very high accuracies, which means the model is not under or over fitting (balanced variance and bias).

Additional visualization of the validation accuracy is analyzed to understand what works or not.

Below is the validation set recall and precision by class.  Note that class 16 has a low recall (false negative), meaning images from class 16 were predicted as some other clases.  In precision chart, class 41 has a low value (false positive).  It is likely that many class 16 images were misclassified as class 41.

![alt text][image9]
![alt text][image10]

Images were pulled from classes 16 and 41 and quickly one can see that some class 16 images have red circular borders are quite faded so they could be similar to class 41 images.  Below are classes 16 (left) and 41 (right) sample images.

![alt text][image11]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image12]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing     | Children crossing  							| 
| Right-of-way          | Right-of-way									|
| Priority road			| Priority road									|
| Turn right ahead 		| Turn right ahead				 				|
| Road work 			| Road work         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%, which is close to 98% from the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the bar charts of the top 5 probabilities for 5 new images.  For all 5 images, they are very close to 100% certainty.  This means the model is really well trained.

Image 1: Children crossing

![alt text][image13]

Image 2: Right-of-way

![alt text][image14]

Image 3: Priority road

![alt text][image15]

Image 4: Turn right ahead

![alt text][image16]

Image 5: Road work

![alt text][image17]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here are some of the visualized feature maps evaluated on the first new image (children crossing).  It seems some feature maps picked up the shape of the triangle.  Some feature maps picked up the shape of the human figures inside the triangle.  Some feature maps picked up the blue sky on the left.

![alt text][image18]
![alt text][image19]
![alt text][image20]
