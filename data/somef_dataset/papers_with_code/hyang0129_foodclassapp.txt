<img src="logo.jpg" align="right" height=125 width=125>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![](https://img.shields.io/github/languages/code-size/hyang0129/food-class-app)
![](https://img.shields.io/github/last-commit/hyang0129/food-class-app)

# Food Classifier 

## Overview 

<img src="appdemo.gif" align="right">

This android app classifies food into one of 101 categories using a cloud compute architecture.

The purpose is to classify different foods in real world settings. 

This project evaluates the suitability of the Google App Engine and various
other technologies for inference in the cloud. 

### App Engine Suitability

The App Engine alone is not suitable for cloud inference. This is because the 
App Engine does not provide significantly greater compute power than a mobile
device. Any ML workload that cannot be run on a mobile device will struggle 
on the App Engine. Additionally, there are major downsides to deploying a 
cloud inference pipeline when edge inference would be acceptable. If the app 
could use the mobile device for inference, it reduces the complexity significantly.

The only realistic use case for the App Engine is to function as a web server
intermediary for cloud inference. In this scenario, a mobile device would 
request a prediction and the App Engine would route the prediction to one 
of many GPU enabled inference servers. 

There does exist a potential use case where a mobile device needs to conserve 
compute resources for whatever reason (save battery, perform other tasks) and
there is insufficient traffic to justify a GPU inference server. However, 
this use case may indicate a problem with the viability of the app itself. 


### Machine Learning Component 

Priority: Good Accuracy, Low FLOPs, Quick Loading

The model used for inference is EfficientNetB3 (EFNB3). 
As shown in https://arxiv.org/abs/1905.11946, EFN architectures provide the 
highest accuracy per flop compared to SOTA at the of publication of 
the paper. B4 was chosen because of its good balance between accuracy and 
flops. Compared to EFNB2, EFNB3 provides a significant boost in accuracy. 
Sizes larger than EFNB4 provide minor increases in accuracy, but significant
increase in flops. EFNB4 was not chosen due to increase in parameters, which
would increase the time to initialize the network. 

#### Dataset Choice 

Food101 https://www.tensorflow.org/datasets/catalog/food101 was chosen 
because of its accessibility and quantity. Note that the training images
have noisy labels. 

#### Model Fine Tuning 

Priority: Train Quickly, Handle Noisy Data, Focus on Hard Examples 

Imagenet pretrained weights were used to initialize the network. Food images
have many common features with imagenet images. To handle noisy data, label 
smoothing https://arxiv.org/pdf/1701.06548.pdf was used. Focal loss was 
chosen over cross entropy in order to increase the impact of hard to classify
examples, even though its primary use is for class imbalance https://arxiv.org/abs/1708.02002.

#### Out of Classes Images 

Currently, the classifier is limited to 100 classes, which means that 
there are many foods that are not included. For any photo that does not
contain one of the 100 classes, the classifier will classify the result
into one of the 100 classes. There is currently no null class. 

### Server Component 

Priority: Low Cost, Quick Scaling 

TFLite was used for quantization of the trained model. Reducing the model footprint
improves the time to start a new instance. To minimize cost, instances are only
active when requests come in, thus it is important to quickly spin up instances 
on the fly. Google App Engine allows for low cost instance management that scales
automatically.  

### App Component 

Android was chosen over iOS due to ease of development and access to android devices.  

## How to Deploy 

In order to deploy the server component, you will need your own Google Cloud 
project and some understanding of the App Engine and Cloud Storage services. 
The follow steps assume that the user has familiarity with those to services. 
If you are new to Google App Engine, I recommend going through the python 3 
tutorial.  

1. Upload your model files (h5 or tflite format) to your GCS bucket. 
2. Update the main.py file to point towards your model file. 
3. Use gcloud app deploy command via the terminal. 

For the android app component, follow the standard steps for a gradle build, 
but be sure to update the URLs in the MainActivity.java file.

If you wish to train and or fine tune the model, see the interactive notebook 
in the training folder. You can copy and run the FOODTRAINER.ipynb in Google Colab
in your web browser with a TPU for free. Be sure to change the model save path to './models/'
to save the models in the Colab instance. If you wish to train the model 
using a Jupyter environment outside of Colab, you will need to allocate the correct
TPU or GPU resource and ensure that the packages are installed correctly. 


## Future Improvements 


### Increased Classes 

If more time and resources were available, it would be possible to improve 
the number of foods for classification through one of two ways. 

1. Increase the number of classes without any changes to the prediction 
pipeline. This method only requires additional data of the new classes and can
be scaled up to 10k+ classes. However, if the number of classes increases beyond
10k the limitations in the final softmax classification layer become more 
problematic. The number of weights in the final layer will eventually exceed
the number of weights in the actual model. Given that the final layer is a 
dense layer with around 1000 inputs, if we had 10k classes, that would require 
10m parameters. The model without the finally layer has around 12m parameters. 

2. Increase the number of classes with a new prediction pipeline. This method 
follows the same pipeline as face recognition and landmark retreival tasks. 
Rather than predict a single class, the model predicts an embedding and then
uses a KNN search to find another image with a similar embedding. This process
allows for the model to train on a large number of classes (say 1k), but then 
predict on classes it has never seem before. This relies on a well calibrated
output embedding that predicts the same embedding values for the same foods, 
but different values for different foods. This method would scale much better
for use cases where we need to classify among 10k+ classes. 

In the second case, it would make sense to implement dedicated 
servers even if the mobile device could perform inference. The necessity is brought by 
the need for the KNN search of the embeddings. It is unrealistic for a mobile
device to store the embeddings of hundreds of thousands of food photos. 

### Null Class 

When a photo is taken of a non food object, the classifier still attempts
to classify it as a food. This can be addressed by adding a new class 
to represent non food and then train the model with this new class. However, 
the variety of non food objects is even greater than the variety of food
objects. There are three potential approaches. 

1. Implement a multi-class approach. This is the most simple as you 
would only need to add another class. However, this can cause problems
during training in terms of the class imbalance. Should non food classes
account for 1%, 10%, or 50% of the training samples? This approach may 
reduce the overall classification accuracy of food classes. 

2. Implement a multi-label approach. This requires two separate outputs.
One represents the original food classes via softmax cross entropy. The 
second is a sigmoid binary cross entropy between is food and not food. 
During training, in examples without any food, there should be no loss
for the softmax cross entropy. This can be implemented by multiplying the 
softmax cross entropy loss per example by the is food label. This approach
should yield better results than the multi-class approach and allows
for easy calibration of the is food threshold because the model outputs
a probability that a image contains a food class. 

3. Implement an object localization and/or segmentation pipeline. This 
can be accomplished via Mask RCNN or similar model. This approach 
is likely to yield the best performance, but it requires bounding boxes
or segmentation maps for the data. This approach makes it more difficult
to scrape images from the web (eg. a simple Google search of boiled eggs
will return a large number of images without masks). For web scraping, 
it may be necessary to utilize unsupervised or semi supervised 
learning methods such as pseudo-labeling. 



 





