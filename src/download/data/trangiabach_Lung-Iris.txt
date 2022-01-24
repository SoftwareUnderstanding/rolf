
https://user-images.githubusercontent.com/62537937/120206800-d6ff5b00-c255-11eb-8074-0297786ebfae.mp4

# Lung-Iris

<img src = "https://github.com/trangiabach/Lung-Iris/blob/main/viral-pneumonai%20(1).jpg?raw=true">

<br/>
Prediction: Pneumonia
<br/>
Red: region of greater interest in classifying pneumonia
<br/>
Blue: region of lower interest in classifying pneumonia
<br/>
This is an image of a pneumonia lungs with expert annotations (arrows) of the pneumonia regions. As can be seen, the AI have detected correctly the region of interest (red regions) that have been annotated by the expert.


## Demo

Link: https://lung-iris.herokuapp.com/

Note: Due to the spontaneous nature of Heroku web hosting, there could be delays and slow load time. Please reload if necessary.

## Introduction

Lung Iris is a web application that can automatically detect pneumonia patients from their chest X-ray images. With more than 2 billions procedures for pneumonia diagnosis performed manually by radiologists worldwide, Lung Iris serves as the optimal solution for saving time and cost for the healthcare system. In addition, in view of the COVID-19 pandemic, Lung Iris can help identify critical COVID-19 patients who have developed pneumonia in their lungs, making the diagnosis time much faster for them so that they can receive prompt care. This is particularly effective in situations where the healcare system is overloaded and there is a lack of personnel.


## Inspiration


Diagnosis from medical imaging is very common and it is estimated that there are around 2 billion procedures performed worldwide. The case is even more evident when in Vietname, most of X-ray scans are for the lungs and heart. Therefore, automating this process will be of great significance.

COVID-19 is still continuing to spread around the world at alarming rates. What will happen if the healthcare system is overloaded? How will we be able to take off burden from already extremely tired medical staff? And especially, how can we help critical patients in whose situation time is extremely valuable?


A survey by the University of Michigan shows that patients have an average wait time for chest X-ray results from 2-3 days. This number can also be as high as 11 days. However, with AI technology applied, this wait time can be reduced to less than one day.


## What it does

The web application can detect whether a chest X-ray scan image is pneumonia-positive or normal. It also give percentages of the two labels and highlight regions of interest.

## How it is built

### Dataset

The dataset is synthesized from 3 Kaggle datasets comprising of pneumonia and normal chest X-ray images, consisting of more than 18000 training data points, 5000 validation data points and 614 testing data points. Types of pneumonia include viral, bacterial and COVID-19 pneumonia. The dataset was then preprocessed and augmented with random flips, zooms, sheers and rotations. An additional thresholding algorithm from OpenCV was also applied to hightlight important features in the chest X-rays.

### Model

The model is a convolutional neural network comprising of four convolutinal layers each with max-pooling and two fully-connected layers. An activation map was extracted from the last convolutional layer, allowing for the network to have great localization ability. The model achieved 96% accuracy on training and validation set and 91% accuracy on testing set.

### Web application

The model is configured with a Flask backend, in which an API is installed to make predictions and draw out the activation map. The front-end is comprised of HTML, CSS and jQuery. The app is then deployed and hosted on Heroku server. 


### Technologies used

- Tensorflow
- Keras
- Flask
- jQuery
- PIL
- OpenCV

## Challenges

One of the biggest challenge was how to create a small model that can attain great accuracy. Transfer learning can be a great way to attain accuracy fast but due to the relative large size of the model (often more than ten million parameters and >200MB in size), it is very difficult to deploy on small servers. The most optimal model was achieved only after a great deal of time fine-tuning hyper parameters and datasets preprocessing.

## Accomplishments

The resulting model attained good accuracy and only have around two million paramter and <30MB in size, making it less prone to overfitting, scalable and fast to deploy on servers. The model also has great localization abilities (highlight regions of interest) through the use of activation maps extracted from the convolutional layers.

## Lessons

A great deal was learnt about how to optimize model for best performance and how to configure AI in a production environment. This project can not be realized without the help of friends who acted as advisors and researchers, engineers who have kindly shared their knowledge to the world. 

## Updates

1. Collaborate with researchers and radiologist to collect better data.
2. Find hospital to test the solutions.
3. Add bounding box on top of the activation map to further localize suspected areas. A model with multi-labels of more 14 thoracic diseases is also currently being developed from another dataset from VinBigData. With new addition of labels for diseases will mean that radiologists can use the tool for an even more generalized purpose of identifying diseases from chest X-rays. 
4. Integrate more variables such as age, gender, sex, weight,... into the computer vision model to make diagnosis even more detailed.
5. Integrate hospital specific procedures to be more applicable such as face mask detection and employee/ medical staff face recognition to gain access to the web app.

## References

Radiologist want patients to have results faster:
https://www.reuters.com/article/us-radiology-results-timeliness/radiologists-want-patients-to-get-test-results-faster-idUSKBN1DH2R6

Transfer learning with chest X-rays:
https://www.nature.com/articles/s41598-020-78060-4

CheXnet: 121-layer Convolutional neural network for identifying pneumonia:
https://arxiv.org/pdf/1711.05225.pdf

Localization using bounding box: 
https://ml4health.github.io/2019/pdf/73_ml4h_preprint.pdf







