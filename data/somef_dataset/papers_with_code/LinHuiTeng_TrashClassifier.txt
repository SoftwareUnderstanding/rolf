# TrashClassifier
## Overview
This is an example application for TensorFlow Lite on Android. It classifies whatever it sees on the device's camera into 5 kinds of recyclable garbage in real-time.
## Model
Add custom FC layer as an new output in a pre-trained and optimized model named MobileNet. This fine-tuned model is re-trained with the dataset from Kaggle's garbage-classification challenge. 
## Deploy
The model was first converted to tensorflow-lite format and put into the image classified application example to replace the original model.

MobileNet : https://arxiv.org/pdf/1704.04861.pdf<br>
Dataset : https://www.kaggle.com/asdasdasasdas/garbage-classification  <br> 
Android application example : https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md
