# TensorFlow Android Camera Demo

This folder contains an example application utilizing TensorFlow for Android
devices.

## Description

The demos in this folder are designed to give straightforward samples of using
TensorFlow in mobile applications.

Inference is done using the [TensorFlow Android Inference Interface](../../../tensorflow/contrib/android),
which may be built separately if you want a standalone library to drop into your
existing application. Object tracking and YUV -> RGB conversion is handled by
libtensorflow_demo.so.

A device running Android 5.0 (API 21) or higher is required to run the demo due
to the use of the camera2 API, although the native libraries themselves can run
on API >= 14 devices.

## Current samples:

1. [TF Classify](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/ClassifierActivity.java):
        Uses the [Google Inception](https://arxiv.org/abs/1409.4842)
        model to classify camera frames in real-time, displaying the top results
        in an overlay on the camera image.
2. [TF Detect](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/DetectorActivity.java):
        Demonstrates a model based on [Scalable Object Detection
        using Deep Neural Networks](https://arxiv.org/abs/1312.2249) to
        localize and track people in the camera preview in real-time.
3. [TF Stylize](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/StylizeActivity.java):
        Uses a model based on [A Learned Representation For Artistic Style]
        (https://arxiv.org/abs/1610.07629) to restyle the camera preview image
        to that of a number of different artists.

<img src="sample_images/AgriHome.jpg" width="30%"><img src="sample_images/BrownSpot.jpg" width="30%"><img src="sample_images/RedStripe.jpg" width="30%">