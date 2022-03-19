# YOLOv3-TF2
## Introduction
This is my implementation of YOLOv3 using TensorFlow 2.0 backend. The main purpose of this project is to get me familiar with deep learning and specific concepts in domain object detection. Two usages are provided:
* Object detection based on official pre-trained weights in COCO
* Object detection of optic nerve on Indian Diabetic Retinopathy Image Dataset (IDRiD) using fine tuning. 
![nerve](/fig/optics_nerve.png)
*Fundus and the corresponding optic nerve*

The following content will be provided in this repo:
* Introduction of YOLOv3
* Object detection based on the official pre-trained weights
* Object detection - fine tuning on IDRiD  



## Introduction of YOLOv3
If you want to go somewhere regarding implementation, please skip this part.  

YOLOv3 is a light-weight but powerful one-stage object detector, which means it regresses the positions of objects and predict the probability of objects directly from the feature maps of CNN. Typical example of one-state detector will be YOLO and SSD series.On the contrary,  two stage detector like R-CNN, Fast R-CNN and Faster R-CNN may include
Selective Search, Support Vector Machine (SVM) and Region Proposal Network (RPN) besides CNN. Two-stage detectors will be sightly more accurate but much slower.
 
YOLOv3 consists of 2 parts: feature extractor and detector. Feature extractor is a Darknet-53 without its fully connected layer, which is originally designed for classification task on ImageNet dataset.   
![darknet](/fig/Darknet.png)  
*Darknet-53 architecture(Source: YOLOv3: An Incremental Improvement https://arxiv.org/abs/1804.02767)*

Detector uses multi-scale fused features to predict the position and the class of the corresponding object.  
![yolov3](/fig/yolo.png)*(Source: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)*

As you can see from the picture above, there are 3 prediction scales in total. For example, if the spatial resolution of an input image is 32N X 32N, the output of the first prediction convolution layer(strides32) will be N X N X (B X (C+5)). B indicates amount of anchors at this scale and C stands for probabilities of different classes. 5 represents 5 different regressions, the  horizontal offset t_x, the vertical offset t_y, resizing factor of the given anchor height t_hand width t_wand objectness score o (whether an object exists in this square of the checkerboard). The second prediction layer will output feature maps of 2N X 2N X (B X (C+5)). And the third prediction output will be much finer, which is 4N X 4N X (B X (C+5).

Reading papers of YOLO, YOLOv2 and YOLOv3, I summarize the loss function of YOLOv3 as follows:  
![](/fig/loss1.PNG)
<!-- $$
L_{Localization} = \lambda_1\sum_{i=0}^{N^2}\sum_{j=0}^{B}1_{ij}^{obj}[(t_{x} - t_{\hat{x}})^2 + (t_{y} - t_{\hat{y}})^2]
\\L_{Shaping} =\lambda_2\sum_{i=0}^{N^2}\sum_{j=0}^{B}1_{ij}^{obj}[(t_{w} - t_{\hat{w}})^2 + (t_{h} - t_{\hat{h}})^2]\\
L_{objectness-obj} =\lambda_3\sum_{i=0}^{N^2}\sum_{j=0}^{B}1_{ij}^{obj}\log(o_{ij})$$ $$L_{objectness-noobj} =\lambda_4\sum_{i=0}^{N^2}\sum_{j=0}^{B}1_{ij}^{obj}\log(1-o_{ij})
\\L_{class} =\lambda_5\sum_{i=0}^{N^2}\sum_{j=0}^{B}1_{ij}^{obj}\sum_{c\in classes}[p_{\hat{ij}}(c)\log(p_{ij}(c))+ (1-p_{\hat{ij}}(c))\log(1-p_{ij}(c))])
\\ L_{Scale_{1}} = L_{Localization} + L_{Shaping} + L_{objectness-obj} + L_{objectness-noobj} + L_{class}
\\ L_{total} = L_{Scale_{1}}+L_{Scale_{2}}+L_{Scale_{3}}$$ -->

## Object detection based on the official pre-trained weights
In this part, object detection could be executed on videos or camera input.
### Dependencies
 * Python 3.6
 * Opencv-python 4.1.2.30
 * TensorFlow 2.0.0
 * Numpy 1.17.3
 * Seaborn 0.10.0
### Usage
#### Execution
Please download the official pre-trained weights on COCO dataset and put the weights file under the root directory of this project.  
https://pjreddie.com/media/files/yolov3.weights   
In the terminal, enter ```python3 video_detect.py``` to execute video object detection task using pre-trained weights.
#### Flags
* Video: If true, the specific video will be processed. Else, the corresponding camera will be called to record video.
* Video_path: Specify the path of video. Valid only when video option is true.
* Output_folder: Video after processing will be output in this path.
* Model_size: Video frames will be resized to this size and be put into CNN model, higher resolution leads to more accurate detection (especially for small objects) and slower speed.
* Iou_threshold: Threshold of non-max suppression.
* Max_out_size: maximum amount of objects in one class
* Confid_threshold: neglect the detected objects under the certain confidence. 
### Inference Performance
```@tf.function``` is enabled by default to improve performance, namely, no eager execution. No batching is applied here. Performance is measured on platform Intel i7 9750H, **GTX 1660 Ti 6GB**, DDR4 2666 8GB*2, Sabrent Rocket NVME 2TB
Model size | Average FPS
------------ | -------------
320*320 | 30.1
416*416 | 22.3
608*608 | 13.9

### Demo Video
[![](http://img.youtube.com/vi/6mWNgng6CfY/0.jpg)](http://www.youtube.com/watch?v=6mWNgng6CfY "")   
https://www.youtube.com/watch?v=6mWNgng6CfY&t=3s
## Object detection - fine tuning on IDRiD
### Features
* TFRecord: efficient data loading
* @tf.function: efficient training (Building graph)
* New anchor priors from K-Means++
* Data augmentation
* Checkpoint autosave
### Usage
#### Execution
Please download the official pre-trained weights on COCO dataset and put the weights file under the root directory of this project.  
https://pjreddie.com/media/files/yolov3.weights
First, enter ```python3 k-means.py``` to generate new anchors on our IDRiD dataset. Copy the new anchors into ```trainer_main.py```. In the terminal, enter ```python3 trainer_main.py``` to begin fine tuning.
##### Training using Provided Checkpoints
Please download the checkpoint files from https://www.dropbox.com/s/nb8q5b8a8lkcor3/tf_ckpts.tgz?dl=0. Decompress the content into the folder ```tf_ckpt```, training will be restored automatically.
#### Flags
Except for part of the previous flags, there might be the following flags needing to be noticed.
* lr: learning rate of the Adam optimizer, by default 10e-3 for fine tuning. If full transfer learning(including feature extractor) is needed, please further reduce the learning rate.
* epoch: Training will be stopped only until this number.
* finetuning: When True, variables in feature extractor, including those in batch normalization will be frozen.
* load_full_weights: True only when your training dataset has a same mount of classes as COCO when you would like to fine tune your DNN on this dataset.
### Implementation Details
#### Image Labeling
I manually annotate the positions of optic nerve using labelImg https://github.com/tzutalin/labelImg. For each image, a XML file with positional information will be attached. There are 413 images for training set and 103 images for testing.
#### XML TO CSV
A CSV file to summarize all labels and positional information will be efficient for the following processing. Here I use the conversion from https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py.
#### Pre-processing and TFRecord
The original image size is 4288 X 2848. The images are cropped and padded to ratio 1:1. After that, they are all resized to 608 X 608 and written into TFRecord. In this way, data could be loaded efficiently.
#### Clustering using K-Means++
From the original papers of YOLOv3 and its predecessor YOLOv2, one of the trick to gain more performance on a specific dataset is to use a better set of anchors priors. Since our dataset is a customized dataset different from COCO. It might be beneficial to use the new priors. Here I use K-Means++ instead of K-Means to make clustering less sensitive to initialization. For practice, I don't use any existing package but implement the algorithm by myself. I tried two different setups, one is to generate 9 different sizes of anchors, another is to generate 3 anchors. Results with non-changed COCO anchors will also be reported later.
Name | anchors
------------ | -------------
COCO original | [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
IDRiD-9 (threshold=0.99) | [(77, 92), (89, 93), (83, 101), (95, 101), (93, 111), (105, 109), (101, 119), (114, 125), (151, 152)]
IDRiD-9 (threshold=0.98) | [(77, 91), (89, 93), (83, 101), (95, 102), (92, 111), (104, 108), (98, 117), (110, 122.0), (127, 134)]
IDRiD-3 | [(86, 98), (97, 108), (110, 120)]
##### Distance of K-Means Clusters
Following YOLOv2:  
![](/fig/IOU.png)
##### Flags
* NUM_CLUSTER: how many clusters will be built.
* NEW_CENTRIOD_THRESHOLD: K-Means is sensitive to initialization. Theoretically, it should be set as 1.0. In this situation, the furthest point will be chosen as the centriod of the new cluster at the initialization. Pro: larger coverage of data points;  con: sensitive to outliers. By default, threshold is set as 0.98.
##### Demo of K-Means++
![](/fig/k-means.gif)
#### Data augmentation
Since our dataset is a rather small dataset, after train-validation-split, there are about 360 images for training, I use data augmentation to mitigate over-fitting. Data augmentation contains: 
* random brightness
* random contrast
* random hue  
Further data augmentation methods could be considered: as flipping, shearing and shifting.
##### Flags
* probability: percentages of augmented images.
#### Training and checkpoints
Before training, all pre-trained weights except for those for the last 3 detection layers will be loaded. The weights of feature extractor will be frozen. I also tried unfreezing feature extractor to support full transfer learning. Unfortunately, my VRAM (6GB) is limited and unable to finish the training. During the training, checkpoints will be saved under the directory ```./tf_ckpt``` automatically every 5 epochs. If there is any valid checkpoint file in this folder, training will be restored.
#### Training Monitor and Metrics
In order to monitor the training, TensorBoard is set up in the folder ```results```. To view the result, please enter```tensorboard --logdir results``` under the corresponding project directory. Intersection over Union (IoU) and sum loss are the metrics to monitor the training.

### Final results

Learning rate | anchors | data augmentation | Final IOU on IDRiD(1250 iterations)
------------ | -------------|-------------|-------------
10e-2 | COCO | NO | 62.96%
10e-3 | COCO | NO | 65.41%
10e-4 | COCO|NO| 0.00%(not succeeded)
10e-3 | IDRiD-9 (Threshold=0.99) | NO | 66.29%
10e-3 | IDRiD-9 (Threshold=0.98) | NO | **67.78%**
10e-3 | IDRiD-3 | NO | 67.27%
10e-3 | IDRiD-9 (Threshold=0.98)| Yes (Probability=0.5) | 67.53%
10e-3 | IDRiD-9 (Threshold=0.98)| Yes (Probability=0.8) | 66.83% / **67.69% (2500 iterations)**
#### Example
What in the red frame line is our ground truth.
![](/fig/detection1.gif)
