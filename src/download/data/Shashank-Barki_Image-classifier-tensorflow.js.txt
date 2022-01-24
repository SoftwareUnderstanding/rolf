# Image-classifier-tensorflow.js
Image-classifier using tensorflow.js

Requirements
--------------

A recent version of Chrome or another modern browser.
A text editor, either running locally on your machine or on the web via something like Codepen or Glitch.
Knowledge of HTML, CSS, JavaScript !
Laptop with webcam enabled

Contents
----------

image.html
image.js


What are we doing ?
--------------------

I have built a custom image classifier using the webcam on the fly.
I have made a classification through MobileNet with internal representation (activation) of the model for a particular webcam image and used that for classification.


How ?
-------------------

MobileNets and tensorflow.js for training a model called K-nearest neighbours

MobileNets is used for mobile and embedded vision applications. 
MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.


TensorFlow.js, an open-source library you can use to define, train, and run machine learning models entirely in the browser, using Javascript and a high-level layers API. 
If you’re a Javascript developer who’s new to ML, TensorFlow.js is a great way to begin learning. 
Or, if you’re a ML developer who’s new to Javascript, read on to learn more about new opportunities for in-browser ML. 

I have used a module called a "K-Nearest Neighbors Classifier", which effectively lets us put webcam images (actually, their MobileNet activations) 
into different categories (or "classes"), and when the user asks to make a prediction we simply choose the class that has the most similar activation 
to the one we are making a prediction for.

Now when you load the index.html page, you can use common objects or face/body gestures to capture images for each of the three classes. 
Each time you click one of the "Add" buttons, one image is added to that class as a training example. 
While you do this, the model continues to make predictions on webcam images coming in and shows the results in real-time.

How to run this ?
--------------------

Create a folder in local laptop and save image.html and image.js
Enable webcam from your laptop
Click image.html for real time webcam image classification.

Source:

https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db
https://arxiv.org/abs/1704.04861?source=post_page---------------------------
https://www.tensorflow.org/js/tutorials/transfer/image_classification 
https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
