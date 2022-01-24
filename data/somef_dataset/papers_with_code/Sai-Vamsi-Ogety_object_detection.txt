##  Object Detection
- Object detection is the act of finding the location of an object in an image
- Image classification labels the image as a whole. Finding the position of the object in addition to labeling the object is called object localization. 
- Typically, the position of the object is defined by rectangular coordinates. 
- Finding multiple objects in the image with rectangular coordinates is called detection.
![Object Detection](https://upload.wikimedia.org/wikipedia/commons/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)

### Datasets for Object Detection:
- ImageNet
- PASCAL VOC
- COCO

### Metrics used to evaluatet these Datasets:
- Intersection over Union (IoU) : The IoU is the ratio of the overlapping area of ground truth and predicted area to the total area
- Mean Precision Average (mAP) : The mAP metric is the product of precision and recall of the detected bounding boxes. The mAP value ranges from 0 to 100. The higher the number, the better it is.

### Approaches :
## Object Detection as Regression :

![](Images/reg.JPG)

## Object Detection as classification(Sliding Window) :

![](Images/slid.JPG)

## Region Proposal :

![](Images/regpro.JPG)

## Family of R-CNN Methods (Region Based Methods for Object Detection):

### 1. Regions of the convolutional neural network (R-CNN) :

- The first work in this series was regions for CNNs proposed by Girshick et al.(https://arxiv.org/pdf/1311.2524.pdf) . 
- It proposes a few boxes and checks whether any of the boxes correspond to the ground truth. Selective search was used for these region proposals. 
- Selective search proposes the regions by grouping the color/texture of windows of various sizes. The selective search looks for blob-like structures. 
- It starts with a pixel and produces a blob at a higher scale. It produces around 2,000 region proposals. 
- This region proposal is less when compared to all the sliding windows possible.

## Step 1 :

![](Images/rcnn1.JPG)

## Step 2:

![](Images/rcnn2.JPG)

## Step 3:

![](Images/rcnn3.JPG)

## Step 4:

![](Images/rcnn4.JPG)

## Step 5:

![](Images/rcnn5.JPG)

## Step 6:

![](Images/rcnn6.JPG)

## Problems with R-CNN :
- Training is slow (84h), takes a lot of disk space
- Inference (detection) is slow 47s / image with VGG16 [Simonyan & Zisserman. ICLR15]

--- 
### 2. Fast R-CNN :

## Step 1: 

![](Images/frcnn1.JPG)

## Step 2: 

![](Images/frcnn2.JPG)

## Step 3: 

![](Images/frcnn3.JPG)

## Step 4: 

![](Images/frcnn4.JPG)

## Results: 

![](Images/frcnn5.JPG)

---
### 3. Faster R-CNN:

## Step 1: We learn to predict those Region Proposals than passing through traditional algorithm

![](Images/frcnn1.JPG)

## Result :

![](Images/frcnn2.JPG)

--- 
## Non-Region based methods or Detection without Proposals :

- One of the characterestics of this method is that there is no per region processing like the one that happens in Region based methods but rather 
the processing happens on the entire image from start to end. And this sorts of improves speed a lot.

### YOLO (You Only Look Once) / SSD (Single Shot Detection) :

![](Images/yolo1.JPG)

### Hyperparamters :

![](Images/yolo2.JPG)










## New markdown cell
