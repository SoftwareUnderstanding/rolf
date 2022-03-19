# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/histogram_training.png "Histogram of training data"
[image2]: ./writeup_images/histogram_valid.png "Histogram of validation data"
[image3]: ./writeup_images/mean_std.png "Mean and standard deviation of data"
[image4]: ./writeup_images/Equalization.png "Equalization techniques considered"
[image5]: ./writeup_images/Problem_1.png "Children crossing"
[image6]: ./writeup_images/Problem_2.png "Bumpy road"
[image7]: ./writeup_images/internet_images.png "Internet images"
[image8]: ./writeup_images/softmax1.png "Softmax 1"
[image9]: ./writeup_images/softmax2.png "Softmax 2"
[image10]: ./writeup_images/softmax3.png "Softmax 3"
[image11]: ./writeup_images/softmax4.png "Softmax 4"
[image12]: ./writeup_images/softmax5.png "Softmax 5"
[image13]: ./writeup_images/softmax6.png "Softmax 6"
[image14]: ./writeup_images/augmentation.png "Augmentation"
[image15]: ./writeup_images/arch.jpg "Architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/utsawk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the shape() property to get the shapes of of training, validation and test datasets. Shape can also be used to find the shape of traffic sign images. Number of classes can be found out using signnames.csv or finding unique entries in the training set - I use the latter

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I plot the normalized histogram of the both the training and validation dataset -  it can be seen that both of the datasets have similar distributions. It can also be seen that some image categories are under-represented like Class # 0 (Speed limit 20 km/hr), Class # 19 (dangerous curve to the left), etc.  

![Histogram of training data][image1]
![Histogram of validation data][image2]

I also plot the mean and standard deviation image. It can be seen from these images that the center of the image carries the traffic sign. The standard deviation is interesting because most of the image is dark - I would have expected the region close to the borders of the image to be varying in pixel intensity because of the varied background of traffic sign images. However, all the images are cropped with traffic sign occupying the majority of the image leading to low standard deviation throughout the 32*32 image.

![Mean and standard deviation of images][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Inspired by [1], I tried two image equalization techniques - histogram equalization and CLAHE (Contrast Limited Adaptive Histogram Equalization) applied to grayscale images [2]. Both these techniques improve the contrast in the image as shown in figure below (figure shows original image, histogram equalized image and CLAHE filtered image from left to right). The 70 in the image is hardly visible in the first image, however, the equalization techniques enhance the image immensely.

![Equalization techniques considered][image4]

I decided to use CLAHE (on grayscale images) for data preprocessing here because histogram equalization does not work well when there are large intensity variations in an image. This is easier to demonstrate on larger images but a couple of examples where histogram equalization does not work well are shown below (as before, figure shows original image, histogram equalized image and CLAHE filtered image from left to right).

![Children crossing][image5]

![Bumpy road][image6]

Additionally, I tried a few data augmentation techniques and ended up using the following augmentations:
* Image rotated randomly in the range +/-[5, 15] degrees and then scaled by 0.9 or 1.1
* Randomly perturbed in both horizontal and vertical directions by [-2, 2] pixels
* Motion blurred with a kernel of size 2

The figure below shows the original RGB image and four processed images used for training (CLAHE filtered grayscale image, scaled and roated, randomly perturbed, and motion blurred)

![augmentation][image14]

Note that the augmentation is applied to grayscaled and CLAHE filtered images. This gives a dataset that is four times the original dataset. Note that each copy of training set image is augmented to produce 4 images and I do not selectively choose certain image categories to augment. Such datasets may represent natural distributions and thus it may not be a good idea to augment unevenly. This is because Augmentation should increase the robustness of the model when seeing unseen images.

I centred the image around the pixel mean and normalized with the standard deviation because I wanted to center the data around zero and have similar ranges for the pixels. Images under different light variations can have largely different pixel values and we desire the network to learn other features in the image than the light conditions, thus centering around the mean and normalization helps the learning process. Normalization also ensures similar values of gradient while doing backpropagation and helps prevent gradient saturation (not too relevant here because image data is already upper bounded).




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x12 	|
| Batch norm     	|  	|
| RELU					|												|
| Max pooling      	| 2x2 stride,  outputs 14x14x12 				|
| Dropout	(a)      	| Keep probability = 0.75 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x32 	|
| Batch norm     	|  	|
| RELU					|												|
| Max pooling      	| 2x2 stride,  outputs 5x5x32 				|
| Dropout	(b)      	| Keep probability = 0.75 				|
| Flatten and Concat (a) (after additional maxpooling) & (b)	| outputs 1388			|
| Dropout     	| Keep probability = 0.75 				|
| Fully connected		| outputs 100      									|
| Batch norm		|       									|
| RELU		|      									|
| Dropout     	| Keep probability = 0.5 				|
| Fully connected		| outputs n_classes (= 43)      									|
| Softmax				|       									|

The overall achitecture is presented in the figure below.
![architecture][image15]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following:
1. Xavier initialization [3]: I saw marked differences in early (in epochs) performance based on the starting weights. When using truncated normal, the training/validation performance was heavily dependent on the mean and standard deviation chosen. The same was true for normal distribution. I used Xavier initiation and immediately saw improvement in early epochs.
2. Batch normalization [4]: I tried batch normalization and saw faster convergence. Even though running it on my computer was taking more time per epoch, batch norm lead to faster convergence (in number of epochs). The exact reasons for batch norm's effectiveness are still poorly understood and is an active area of research [5]. I applied batch normalization before the RELU activation in all the layers, though recently people have been using it post the RELU activation.
3. Regularization: I experimented a lot with dropout probabilities for the different layers and ended up using 0.25 for convolutional layer and concat layer in addition to a dropout of 0.5 for fully connected layers. Without dropout, the network was overfitting easily, which is usually a good sign that the network is implemented correctly.
4. Adaptive learning rate: I tried a few techniques and ended up using a starting learning rate of 0.001 and reducing the learning rate by 0.1 every 20 epochs [6].
5. Batch size of 128. 
6. Adam optimizer: I started with Adam optimizer and it worked well and I did not get a chance to experiment with other optimizers.
7. 100 epochs was used for final submission even though the model seemed to have converged with very few epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.8% 
* test set accuracy of 98.0%


I tried the following architectures: 
1. LeNet (shown in lecture) and enhancements to it including adapting learning rate, dropout for different layers, etc. This is present as a function in my final submission.
2. A VGGNet [7] like architecture (not that deep, but employing same padding level) with 3 convolution layers (convolution+batch norm+RELU+max pooling) and two fully connected layers with adaptive learning rate and dropouts. I excluded this in the final submission .
2. Sermanet architecture shown in [1]. I tried two flavors of it and immediately saw improvement. The main idea here is to "short-circuit" the output of the first convolutional layer directly into the fully connected layer. I saw a marked improvment in the convergence time with this method. The validation accuracy in every run was ~0.97 in just 3 epochs. For the final submission, I let it run for 100 epochs. **The final architecture is based on this and described below. The implementation is SermaNet2() in my submission.**

My journey to submission was long and I spent a lot of time experimenting with hyperparameters:
* I started with the LeNet architecture and tried to study the rest of the design components like data augmentation, weight initialization, learning rate, dropout, batch normalization as described in next few bullets. 
* I started with initial weight optimization study and quickly observed that covergence rate was heavily dependent on initialization hyperparameters of mean/standard deviation and also distribution (truncated gaussian v/s gaussian). I ended up using Xavier initialization after which I never had to worry about weight initialization.
* The second hyperparameter I played with was learning rate. I saw marginal improvement on using learning rate adaptation of reducing it by 0.1 every 20 epochs and continued using it for the rest of the project. I kept a flag to turn off adaptation every now and then to test its effectiveness.
* With the above steps, the model continued to overfit the training data with ~94% accuracy on validation data. I introduced dropout into the model and the validation accuracy improved to ~96%.
* I added batch normalization and it improved convergence rate. I kept a flag and experimented with turning it off and on.
* I wanted to further improve the accuracy and started looking at other architectures like GoogleNet, SermaNet, VGGNet, etc. I implemented SermaNet to the best of my understanding with much smaller feature sizes than in the paper. For example, the paper uses 108-108 filter depth and I used 12-32 filter depth in my submission. I implemented two different flavors of the concatenation layer - one concatenating the second layer with the output of a third convolutional layer and another concatenating output of first and second convolution layer. The latter has lesser parameters and gives better performance and was used for the final submission.
* In the end I tried the VGGNet-like architecture mentioned above, though it gave me slightly lower accuracy than the final submission.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![internet_images][image7]

Other than intentionally picking images that cover the majority of the height and width of the image, I tried to be impartial in selecting the image. The reason I did this was the training data set has images in which traffic sign occupies the majority of the pixel space. I resized the image to 32x32 to fit the modelling. Most images have watermark on them and some of them have varied backgrounds.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Bumpy road     			| Bumpy road 										|
| Slippery road					| Slippery road										|
| Road work	      		| Road work					 				|
| Children crossing			| Children crossing      							|
| Speed limit (60km/h) | Speed limit (60km/h)      |


The model was able to correctly guess 6 out of 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 40th cell of the Ipython notebook.

For most of the images, the softmax probabilities of the correct labels are high, except for the road work sign, which has almost equal softmax probability as the right-of-way at next intersection. This is probably because of the tree being in the background of this particular road sign that confuses the Conv Net.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Right-of-way at the next intersection   									| 
| 0.996     				| Bumpy road 										|
| 0.942					| Slippery road											|
| 0.412	      			| Road work				 				|
| 0.478				    | Children crossing      							|
| 0.426				    | Speed limit (60km/h)     							|

![softmax1][image8] ![softmax2][image9] ![softmax3][image10] 
![softmax4][image11] ![softmax5][image12] ![softmax6][image13]


[1]. http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

[2]. https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

[3]. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

[4]. https://arxiv.org/abs/1502.03167

[5]. https://arxiv.org/abs/1805.11604

[6]. http://cs231n.github.io/neural-networks-3/#anneal

[7]. https://arxiv.org/pdf/1409.1556.pdf


