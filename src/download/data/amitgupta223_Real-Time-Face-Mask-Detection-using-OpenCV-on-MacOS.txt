# Real-Time-Face-Mask-Detection-using-OpenCV-on-MacOS
Going to use OpenCV to do real-time face mask detection from a live stream via my webcam on MacOS.

# How to Access Webcam on MacOS

Create a virtual Environment on MacOS by just opening Terminal and then just run this command: 'source work/bin/activate' to activate virtual environment.
And then put the all necessary file in that work directory to run the python file to access webcam.

python3.7 -m venv work

source work/bin/activate

pip install opencv-python

print ("OpenCV Version:" )

print(cv2.__version__)

Link: https://www.youtube.com/watch?v=nO3csmVyoOQ

# Mask Detection using MobilenetV2

In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of ‘with_mask’ images has made this task more cumbersome and challenging.

# Technology Used

OpenCV: https://opencv.org/

MobileNetV2: https://arxiv.org/abs/1801.04381

Keras: https://keras.io/

TensorFlow: https://www.tensorflow.org/

# Haarcascade classifiers: 

Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.

Link: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

# DataSet: 

This dataset consists of 3835 images belonging to two classes:

with_mask: 1916 images
without_mask: 1919 images
The images used were real images of faces wearing masks. The images were collected from the following sources:

Link: https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG


# Pretrained Model

Link: https://drive.google.com/file/d/16uMH4YwdkA8sdnMlJNE7nv_tBJkX5eNe/view

# Final Output:

[![Watch the video](https://j.gifs.com/GvlZMQ.gif)]

# Credits

https://www.mygreatlearning.com/blog/real-time-face-detection/#sh1

https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/README.md

https://www.kaggle.com/mirzamujtaba/face-mask-detection

https://www.mygreatlearning.com/blog/facial-recognition-using-python/

