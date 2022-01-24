# Face Mask Object Detection

## Executive Summary

### The Problem

The worldwide COVID-19 pandemic has led to the widespread wearing of face masks to protect ourselves and others from spreading the disease. It is particularly important that these be worn when close contact cannot be avoided, and when inside of buildings. My goal is to create an object detection model that can identify whether a person is wearing a mask correctly, incorrectly, or not at all. Here, "correctly" is defined as covering the nose and mouth. This model can be used with both still photos as well as live video feeds.

### Data

Data was sourced from kaggle. The dataset can be found here: https://www.kaggle.com/andrewmvd/face-mask-detection 

There are 853 images included in this dataset. There are a total of 4,072 labeled faces, and of these 2800 faces met the criteria of  > 15 x 15 pixels in size for model training. Classes were imbalanced - 82% of the faces were labeled as "wearing mask", and "mask worn incorrectly" is the least represented class with only 95 trainable faces. Annotations are in PASCAL VOC format.

### Process

To solve this data problem, I take a two-step approach. The first step is building a face mask classifier. I use MobileNet as the base model and train a custom head layer that will separate faces into one of three classes: no mask, mask worn incorrectly, and with mask.

The second step is to run a face detector model to locate all of the faces in an image. Once a face is located, predictions are made using the classifier.

Lastly, these two steps are tied together for assessment on both still images and a live webcam feed.

### Conclusions

The Face Mask Classifier component of the model has a 94% accuracy on both the training and testing data. Categorical cross-entropy was used to measure loss, and after 25 epochs the loss was 0.19. Drilling into the individual classes, "With Mask" performed the best, with a 97% f-1 score and 99% recall - only 6 were mis-classified as without mask during predictions. "Without Mask" also performed well with an f-1 score of 88%.  "Mask Worn Incorrectly" performed most poorly - only 6 of the 29 training samples were predicted accurately. Precision was 100%, but recall was only 21%. With additional samples of incorrectly worn masks, I believe this would perform better.

For face detection, I used two different models: an MTCNN (for images) and an OpenCV built-in DNN (for video).

**MTCNN:** This model is made up of three stages: a proposal network or P-Net that grabs potential candidate windows, followed by a Refine network  or R-Net that calibrates these candidates using bounding box regression, and an Output Network or O-Net that proposes facial landmarks. The MTCNN model detected 2,670 faces, the most out of the three detectors.

**OpenCV DNN:** The OpenCV module comes with a built-in DNN library, and uses a Caffe SSD model built on a Resnet 10 backbone. This model only detected 1,662 faces, but has a much faster performance on video. Predictions were made at 10.8 frames per second.

### Streamlit App

The Streamlit app provides the ability to upload an image and run a face mask detection using the MTCNN model as the detector. This will be a great way to test the generalizability of the model. My hope is to include a webcam version of this model as well but this is under construction at the moment.

### Recommended Applications

**Tracking use of face masks in public spaces:** similar to a traffic counter, this may aid in the study of social behaviors regarding the use of face masks, possibly in dense urban areas with high numbers of pedestrians and airports. 

**Security feature:** One of the benefits of a MobileNet model is that it works well on edge devices. Connected to a locking mechanism, this could be used to prevent non-masked individuals from entering a facility such as a restaurant, retail store, office building, or apartment building. 


### Suggestions for Further Study

**One-Step Detection Model:** As mentioned in the Conclusions section, one of the challenges of a two-step process is the ability to detect a face when it is occluded by a face mask. With a larger dataset to train on, a one-step detection model, such as YOLO or Detectron2, might more effectively detect people wearing face masks.

**Additional Data:** A model is only as good as the data it is trained on. In this instance, this model would benefit specifically from additional images with the "worn incorrectly" classification. It would also be beneficial to bring in a larger range of ethnicities, and photos designed to "trick" the model - such as images of someone with their hand covering their mouth.


## Table of Contents


- Code
  - [01_EDA.ipynb](code/01_EDA.ipynb)
  - [02_face_mask_detector.ipynb](code/02_face_mask_detector.ipynb)
  - [03a_face_mask_detector_opencv.ipynb](code/03a_face_mask_detector_opencv.ipynb)
  - [03b_face_mask_detector_MTCNN.ipynb](code/03b_face_mask_detector_MTCNN.ipynb)
  - [04a_video_mask_detector_opencv.py](code/04a_video_mask_detector_opencv.py)
  - [04b_video_mask_detector_MTCNN.py](code/04a_video_mask_detector_MTCNN.py)
  - [streamlit-app.py](code/streamlit-app.py)
  - face_df.pkl
  - face_mask_detector.h5
  - image_df.pkl
  - opencv_face_detector
    - deploy.prototxt
    - res10_300x300_ssd_iter_140000.caffemodel
- Face Mask Detection - Capstone Presentation


## Sources

##### Data
https://www.kaggle.com/andrewmvd/face-mask-detection  
https://app.roboflow.com/dataset/capstone-face-mask/2  

##### Research 

https://arxiv.org/pdf/2011.02371v1.pdf  
https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/  
https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/about-face-coverings.html  
https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923  
https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470  
https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5  
https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c  

##### Models

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/  
https://arxiv.org/pdf/1801.04381.pdf  

##### Code

https://github.com/opencv/opencv/blob/master/modules/dnn/misc/face_detector_accuracy.py  
https://docs.python.org/3/library/xml.etree.elementtree.html  
https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/  
https://www.kaggle.com/notadithyabhat/face-mask-detector  
https://medium.com/ai-in-plain-english/blood-face-detector-in-python-part-2-building-a-web-application-using-streamlit-in-python-3ff3bde74fe7
