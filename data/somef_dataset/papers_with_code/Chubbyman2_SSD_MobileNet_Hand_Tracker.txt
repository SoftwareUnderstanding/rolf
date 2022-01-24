# SSD_MobileNet_Hand_Tracker
A hand tracker created using OpenCV and an SSD MobileNet v2 re-trained via transfer learning on the EgoHands Dataset.

## Sample Results
<p float="left">
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/gif_demos/hand_tracker_clip_1.gif" height="230" width="413">
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/gif_demos/hand_tracker_clip_2.gif" height="230" width="413">
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/gif_demos/hand_tracker_clip_3.gif" height="230" width="413">
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/gif_demos/hand_tracker_clip_4.gif" height="230" width="413">
</p>

This hand tracker was made as part of my work with the Gesture Detection Project Team at UTMIST. The goal was to utilize the EgoHands Dataset to perform transfer learning on the COCO SSD MobileNet v2, Tensorflow's built-in object detection API. 

## Model Image From SSD Paper
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/miscellaneous/ssd_pic.png">

The EgoHands Dataset, curated by Indiana University, came with a set of labelled annotations that were used to generate TFRecords files, which were required to train the SSD MobileNet. The trained frozen inference graph was then utilized in conjunction with a multithreading approach implemented in OpenCV to detect when a hand was present in a user's webcam input, along with its location on the screen. In the future, this may be implemented to allow the user to interact with their computer's interface, performing actions such as clicking, dragging and dropping, and even playing simple games. 

## Training Results
The final model was trained on 22500 iterations.
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/miscellaneous/mAP_values.PNG">

## Acknowledgements
Code was modified based on the scripts created by Victor Dibia:

https://medium.com/@victor.dibia/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce

https://github.com/victordibia/handtracking

* Note that the original scripts were written in Tensorflow 1.x. Alterations were made accordingly and are present in this repo's code.
However, when using Tensorflow 1.x, include tensorflow.compat.v1 in place of tensorflow.


The EgoHands Dataset can be found here (though downloading it is not required):

http://vision.soic.indiana.edu/projects/egohands/


Harrison @pythonprogramming.net's tutorial on creating the required TFRecords files.

https://pythonprogramming.net/creating-tfrecord-files-tensorflow-object-detection-api-tutorial/


Training code was modified based on the script created by Matus Tanonwong:

https://medium.com/@matus.tanon/custom-object-detection-using-tensorflow-in-google-colab-e4d6e1a17f18


Original Paper Detailing Single Shot Detectors (SSDs) by Liu et al.:

https://arxiv.org/pdf/1512.02325.pdf


Special thanks to Kathy Zhuang for being my partner in this project.

## Certificate
<img src="https://github.com/Chubbyman2/SSD_MobileNet_Hand_Tracker/blob/main/UTMIST%20Certificate.jpg">
