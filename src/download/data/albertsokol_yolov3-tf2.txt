# YOLOv3 in Tensorflow 2.x
An implementation of YOLOv3 from scratch in Tensorflow 2! You can use this to train a YOLOv3 model on any data you like, provided it's in the correct format. Uses the Darknet-53 backbone, and YOLOv3 neck and head as per the paper.

Features:
- Custom generator supports fully customizable image and bounding box augmentation
- AP50 callback - calculate the AP50 at the end of each epoch and save the model with the best performance
- Prediction class for turning images or folders of images into beautiful bounding boxed images 
- Video processing functions for turning videos to frames and vice versa 
- Plenty of documentation and comments so you can easily see what's happening!

Compatible Darknet-53 backbone pre-trained on ImageNet available for download [here](https://drive.google.com/drive/folders/1h9YnOuQO0CRSfrciwZHOhIDGlTzBT7UR?usp=sharing) 

## Contents
- [Demo](#Demo)
- [Requirements](#Requirements)
- [Training](#Training)
    * [pretrain_darknet.py (optional)](#pretrain_darknetpy-optional)
    * [train.py](#trainpy)
    * [Learning rate finder](#Learning-rate-finder)
- [Prediction](#Prediction)
- [References](#References)


## Demo 
[![Demo of model](https://github.com/albertsokol/yolov3-tf2/blob/main/readme_images/youtube_link.png)](https://www.youtube.com/watch?v=tXYPUMHGe7A "My YOLOv3 implementation : object detection on dashcam footage")

## Requirements
- Tensorflow 2.x
- pandas 
- matplotlib
- tqdm (for progress bars) 
- OpenCV (for video processing and pretty bounding boxes)
- imgaug (for image/bbox augmentation)
- imageio (for image i/o)

To create a new anaconda environment with everything you need:

`conda create -n yolov3_env python tensorflow pandas matplotlib tqdm opencv`

`conda activate yolov3_env`

`conda install -c conda-forge imageio imgaug`

## Training
There are 2 different training files: 
 - pretrain_darknet.py
 - train.py


### pretrain_darknet.py (optional)
Use this to pretrain the Darknet-53 backbone on ImageNet/other data before training the full model. Note the provided pre-trained model further up.


### train.py
Train the full YOLOv3 model. 
 - **Required image format**: images may be any dimensions, with training and validation sets in separate folders.
 - **Required input format**: **.csv** file with 7 headers: **filename**, **width**, **height**, **class**, **x1**, **y1**, **x2**, **y2**. **filename** should be the name of the image with extension. **width** and **height** are the image dimensions. **class** is the class of the current bounding box. **x1** and **y1** are the absolute pixel values of the upper-left corner of the bounding box, while **x2** and **y2** are for the bottom-right corner. See train_labels.csv for an example. 

Important training parameters:
 - Edit the train.py file to set `BATCH_SIZE`, and `NUM_EPOCHS`
 - `CNN_INPUT_SIZE` = length and width that the image will be resized to; this is therefore also the input size to the model
 - `raw_anchors` = the dimensions of anchor boxes to use. The provided ones follow the paper 
 - `mode` can be **'lrf'** or **'train'**. See below for more info on learning rate finder. Using **'train'** mode will automatically save the best validation loss model to `save_path`, and best ap50 model to `save_path + ap50`
 - `lr` is the learning rate 

Further options:
 - you can adjust the augmentation parameters by passing in different values for the arguments in the YoloGenerator constructor
 - you can choose whether to freeze the backbone weights or not (will depend on amount of training data you have)
 
Finally, just make sure your paths are set correctly and then run train.py. 

`python train.py`

Note that running the train.py file will create a config.ini file which has the order of classes. It's important this order is the same between training and prediction to ensure you get the right bbox labels. So you might want to save this config.ini file somewhere if you're planning to overwrite it.

### Learning rate finder

The learning rate finder can be activated by setting `mode='lrf'`. 

This mode cycles through all feasible learning rates, and plots the loss against the learning rate. Using this, you can find the optimal learning rate for your configuration. 

Use 1 epoch with 1000-3000 training steps for best results. If the loss explodes without stopping the training, you might have to zoom in a fair bit.

![Image of LRF plot](https://github.com/albertsokol/yolov3-tf2/blob/main/readme_images/lrf_annotated.png)

# Prediction

predict.py reads the config.ini file to get some important info like the anchor sizes, the bounding box labels and their order. Make sure you set the path to the trained model. 

This is the process of taking the large output tensor from YOLOv3 and converting it into a list of bounding box predictions as well as performing non-max suppression, and saving the output images.

A YoloPredictor object is created which has 2 public functions:
 - `predict(fname=None)`: used for debugging and quick checking. Open predict.py in the python shell with the command at the bottom of the file. Run yp.predict() to see how the model performs
 - `predict_folder(folder_paths)`: a list of folder paths to predict on, and save the outputs. Useful for making video frames etc. 
 
If you're using `predict()`, just pass the image directory you want to use to the YoloPredictor constructor. For `predict_folder()` the image directories should be the folder paths given. 

# References
YOLOv3 paper
https://arxiv.org/abs/1804.02767
