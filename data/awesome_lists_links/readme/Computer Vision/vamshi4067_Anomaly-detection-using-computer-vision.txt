![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)
#### Directory Structure
#### scratches_yolo_final.py - Entire code with comments to walk you through the process
## .cfg:
subdivisions = 16
batches = 64
filters = (num of classes + 5) *3
classes = num of classes
## To re calculate anchors
Only if you are an expert in neural detection networks - recalculate anchors for your dataset for width and height from cfg-file: darknet.exe detector calc_anchors data/input_anomaly.data -num_of_clusters 9 -width 416 -height 416 then set the same 9 anchors in each of 3 [yolo]-layers in your cfg-file. But you should change indexes of anchors masks= for each [yolo]-layer, so that 1st-[yolo]-layer has anchors larger than 60x60, 2nd larger than 30x30, 3rd remaining. Also you should change the filters=(classes + 5)*<number of mask> before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.# Darknet #
## If my Repo  doesn't work, use this repo : https://github.com/AlexeyAB/darknet.git
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Yolo v4 paper: https://arxiv.org/abs/2004.10934

Yolo v4 source code: https://github.com/AlexeyAB/darknet

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Dataset Source : http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

## Link to the Colab Notebook to run the model on windows : https://colab.research.google.com/drive/1CE8RJArN37rdFeZzjjBvR5BXHRwYvi3Z#scrollTo=EoisahzAUWWL

#### crazing_1.txt -the sample for creating the annotation.txt file

#### train.txt - example for the train.txt file (the file that contains the absolute path to the images)

#### input_anomaly.data -the file that contains the path for train.txt,test.txt, number of classes,the path to the output folder(backup folder for weights)

#### class.names-then .names file consisting the name of the classes.

#### Text_worked.py - Creates the predicted object class % and the x_center,y_center, width and height of the bounding boxes for the test images

#### Link to the colab notebook to create inputs for the YoloV3 Model : https://colab.research.google.com/drive/1f3WCUqW2YJt_wesRsRWBSDJui-eWnkw6#scrollTo=pbS6IMpaaJ6e

#### Note: the images annotations should be in the same folder if you are using this repo

## Observations:

1)If the image resolution is low, detecting small object is hard.

2)After calculating anchor boxes, still the the small object class detection was not more than 50%

3)Reducing the mini_batch size = batch/subdivisions will increase the computation speed.

4)Weights with highest mAP should be chosen for best results.
## To calculate the mAP use command:
Example : darknet.exe detector recall data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights


6)Increasing the number of anchors , increases the avg IOU and possibility to detect the small classes.But remember to change the masks.
#### Note : Don't do this unless you are an expert in DL or CNN, because this will throw an error.
7)Increasing the max_batches resulted in increase in loss after 12000(num of class = 6, max_batches = num_class * 2000)
####To test on video and capture bbox:
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights test.mp4 -dont_show -ext_output

