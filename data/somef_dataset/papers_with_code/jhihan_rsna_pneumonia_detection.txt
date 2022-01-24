# RSNA Pneumonia Detection Challenge
## Overview
This Kaggle competition task is created on the MD.ai platform in collaboration with the Radiological Society of North America (RSNA) and the American Society of Neuroradiology (ASNR).
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/
This is an object detetion task, in which the classification and the position of desired target objects should be both detected. The position of an object is usually described as a bounding box using four numbers representing upper-left x and y coordinates, along with the height and width of the box.

## Mask RCNN
In deep learning, convolutional neural network (CNN) is a class of deep neural networks, most commonly applied to analyzing image datas. Region-based convolutional neural networks (RCNN) combines rectangular region proposals, which are used to predict the position of bounding boxes, and the CNN features. There are some other modified methods: Fast RCNN, Faster RCNN, Mask RCNN...etc.

Mask RCNN contains consists of two stages. The first stages scans the image and generates proposals(areas likely to contain an object). And the second stage in parallel classifies the proposals and generates bounding boxes and masks.

In addition to the existing branch for classification and bounding box regression in the original RCNN, Mask RCNN adds a branch for predicting segmentation masks on each Region of Interest (RoI) (Figure \ref{MRCNN_framework}). The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-topixel manner.
![Image of Mask R-CNN framework\label{MRCNN_framework}](https://github.com/jhihan/rsna_pneumonia_detection/blob/master/images/mask_rcnn_framework.png)
*The Mask R-CNN framework for instance segmentation. Source: https://arxiv.org/abs/1703.06870*

## Transfer Learning
Humans have an ability to transfer knowledge across tasks. The idea of transfer learning is to use the learned features in other problems to learn a new task. Some low level features from A could be helpful for learning B. For example, in the problems of the computer vision, certain low-level features, such as edges, shapes, corners and intensity, can be shared across tasks, and thus enable knowledge transfer among tasks. Deep learning models are good candidates to handle transfer learning because these models are layered architectures that learn different features at different layers. These layers are then finally connected to a last layer to get output. Therefore, we can utilize a pre-trained network without its final layer as a fixed feature extractor for other tasks.

There are lots of stored pretrained weights of particular models from large database. We can use these pretrained weights use them as a starting point for further training even though the training targets are totally different. In this project, we will start from the pre-trained COCO weights, which is provided by the auther of the Mask RCNN package we will use.

## Requirements of Mask RCNN package
Python 3.4, TensorFlow 1.3 (but < 2.0), Keras 2.0.8 and other common packages listed in requirements.txt of Mask RCNN packages developed by Waleed Abdulla: https://github.com/matterport/Mask_RCNN.
## Installation
Follow the instruction in Waleed Abdulla's Github page.


