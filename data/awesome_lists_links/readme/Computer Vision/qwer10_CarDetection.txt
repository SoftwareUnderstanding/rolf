# CarDetection

## Descriptioin
This project will detect cars in one image using the YOLO model. 
Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242).

Because the YOLO model is very computationally expensive to train, I will load pre-trained weights to use.

In this project I used the dataset from [drive.ai](https://www.drive.ai/)

Two main work have done：

1. Use object detection on a car detection dataset
2. Deal with bounding boxes

### Model
the architecture as follows:
![image](https://github.com/qwer10/CarDetection/blob/master/nb_images/architecture.png)


## Requirements
1. TensorFlow 
2. python 3 or later
3. python packages : Keras, numpy, scipy, pandas, PIL
4. download the weights yolo.h5 on the [website](https://pjreddie.com/darknet/yolo/)


## Usage
```
python cardetection.py
```
## Resualts
the resuals as follows:
![image](https://github.com/qwer10/CarDetection/blob/master/out/test.jpg)

