# Object Detection

We will learn about object detection using the very powerful YOLO model. Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242). 

**We will learn to**:
- Use object detection on a car detection dataset
- Deal with bounding boxes

## 1 - Problem Statement
You are working on a self-driving car. As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around. 
 <p align="center"><img src="nb_images/1.gif" style="width:750px;height:200px;"></p>

<p align="center"><strong> Pictures taken from a car-mounted camera while driving around Silicon Valley. <br> We would like to especially thank <a href="https://www.drive.ai/">drive.ai</a> for providing this dataset! Drive.ai is a company building the brains of self-driving vehicles.</strong></p>

 <p align="center"> <img src="nb_images/driveai.png" style="width:100px;height:100;"> </p>

You've gathered all these images into a folder and have labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like.

<img src="nb_images/box_label.png" style="width:500px;height:250;">
<p align="center"><strong><u>Figure 1</u>: Definition of a box</strong><p>

If you have 80 classes that you want YOLO to recognize, you can represent the class label **c** either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0. 

In this exercise, we will learn how YOLO works, then apply it to car detection. Because the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use. 

## 2 - YOLO
### 2.1 - Model details

First things to know:
- The **input** is a batch of images of shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers **(p<sub>c</sub>, b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>, c)** as explained above. If we expand **c** into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

We will use 5 anchor boxes. So we can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

Lets look in greater detail at what this encoding represents. 

<img src="nb_images/architecture.png" style="width:700px;height:400;">
<p align="center"><strong><u>Figure 2</u>: Encoding architecture for YOLO</strong></p>

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

<img src="nb_images/flatten.png" style="width:700px;height:400;">
<p align="center"><strong><u>Figure 3</u>: Flattening the last two last dimensions</strong></p>

Now, for each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class.

<img src="nb_images/probability_extraction.png" style="width:700px;height:400;">
<p align="center"><strong><u>Figure 4</u>: Find the class detected by each box</strong></p>

Here's one way to visualize what YOLO is predicting on an image:
- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across both the 5 anchor boxes and across different classes). 
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

<img src="nb_images/proba_map.png" style="width:300px;height:300;">
<p align="center"><strong><u>Figure 5</u>: Each of the 19x19 grid cells colored according to which class has the largest predicted probability in that cell.</strong></p>

Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm. 

Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

<img src="nb_images/anchor_map.png" style="width:200px;height:200;">
<p align="center"><strong><u>Figure 6</u>: Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes.</strong></p>

In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. You'd like to filter the algorithm's output down to a much smaller number of detected objects. To do so, we'll use non-max suppression. Specifically, we'll carry out these steps: 
- Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class)
- Select only one box when several boxes overlap with each other and detect the same object.

### 2.2 - Filtering with a threshold on class scores

We are going to apply a first filter by thresholding. We would like to get rid of any box for which the class "score" is less than a chosen threshold. 

The model gives us a total of 19x19x5x85 numbers, with each box described by 85 numbers. It'll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape **(19 X 19, 5, 1)** containing **p<sub>c</sub>** (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape **(19 X 19, 5, 4)** containing **(b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>)** for each of the 5 boxes per cell.
- `box_class_probs`: tensor of shape **(19 X 19, 5, 80)** containing the detection probabilities **(c<sub>1</sub>, c<sub>2</sub>, ... c<sub>80</sub>)** for each of the 80 classes for each of the 5 boxes per cell.

### 2.3 - Non-max suppression 

Even after filtering by thresholding over the classes scores, we still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

<img src="nb_images/non-max-suppression.png" style="width:500px;height:400;">
<p align="center"><strong>Figure 7: In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probabiliy) one of the 3 boxes.</strong></p>

Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.
<img src="nb_images/iou.png" style="width:500px;height:400;">
<p align="center"><strong>Figure 8: Definition of "Intersection over Union".</strong></p>

We are now ready to implement non-max suppression. The key steps are: 
1. Select the box that has the highest score.
2. Compute its overlap with all other boxes, and remove boxes that overlap it more than `iou_threshold`.
3. Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

### 2.4 Wrapping up the filtering

It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented. 

`yolo_eval()` - takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail we have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which we have provided): 

```python
boxes = yolo_boxes_to_corners(box_xy, box_wh) 
```
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of `yolo_filter_boxes`
```python
boxes = scale_boxes(boxes, image_shape)
```
YOLO's network was trained to run on 608x608 images. If we are testing this data on a different size image--for example, the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.

**Summary for YOLO**:
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers. 
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
    - 85 = 5 + 80 where 5 is because **(p<sub>c</sub>, b<sub>x</sub>, b<sub>y</sub>, b<sub>h</sub>, b<sub>w</sub>, c)** has 5 numbers, and and **80** is the number of classes we'd like to detect
- We then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives the YOLO's final output. 

## 3 - Test YOLO pretrained model on images
In this part, we are going to use a pretrained model and test it on the car detection dataset.

### 3.1 - Defining classes, anchors
Recall that we are trying to detect 80 classes, and are using 5 anchor boxes. We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" and "yolo_anchors.txt".

### 3.2 - Loading a pretrained model

Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. We are going to load an existing pretrained Keras YOLO model stored in "y1.h5". (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. Technically, these are the parameters from the "YOLOv2" model, but we will more simply refer to it as "YOLO" in this notebook.)


**What we should remember**:
- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. 
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically: 
    - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise. 

***
Here are few examples, 

<table>
<td> 
<img src="images/test5.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/test5.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>
<table>
<td> 
<img src="images/test2.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/test2.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>
<table>
<td> 
<img src="images/dog.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/dog.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>
<table>
<td> 
<img src="images/check5.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/check5.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>
<table>
<td> 
<img src="images/test4.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/test4.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>
<table>
<td> 
<img src="images/firedog.jpg" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="out/firedog.jpg" style="width:500;height:500px;"> <br>
</td> 
</table>

***
**References**: 
The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 

**Car detection dataset**:
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. We are especially grateful to Brody Huval, Chih Hu and Rahul Patel for collecting and providing this dataset. 
