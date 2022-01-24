# Binary YOLOv2
This is yolov2 implementation in pytorch.
Original paper - [arxiv](https://arxiv.org/abs/1612.08242)
![YOLO target functionallity. Not current results](https://i.ytimg.com/vi/QCX2DLhZS0M/maxresdefault.jpg)

## Intro
### Purpose
- This was project for course Biometrics2. The target was to detect faces on image and then apply other net to categorize them. In yolonet object detection and recognition is done in one step but this was opportunity to implement binary yolo.
### Dataset
- Dataset image with faces and location. On one image could be one, many or none.
### Goal
- Find face(s) on image and return coordinates.


## Differences from original paper
- One class vectors - face, no face. There could be changed vector of probabilities per frame. In their paper the vector is [p, bx, by, bw, bh, p1, p2, ...] where p is certainity of centre of object, bx,by,bw,bh bounding box properties and p_i is class probability. As there is only one class the vector is only [p, bx, by, bw, bh]. The loss function is changed accordingly.
- Model: For training on my one gpu and to speedup process the network is adapted resnet. The last 4 layers of resnet is cut and then added few convolution layers wit leaky rely activations. Also there is implemented parsing yolov3.cfg to build origianl model yolov3 and use transfer learning to tune parameters but training was not tested.

## Prerequisites
- Python 3.6 >
- Dataset with box annotation

### Dataset
Structure:

+  dataset/
    +   class1/
    +   class2/
    +   annotation.csv

In annotation.csv is csv formatted file:

filepath | box_x1 | box_x2 | box_y1 | box_y2
-------- | ------ | ------ |------ | ------
kękę/63.jpg | 93 | 649 | 179 | 563