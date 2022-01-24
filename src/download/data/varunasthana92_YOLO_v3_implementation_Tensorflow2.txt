# YOLO_V3 object detection implementation in tensorflow2 using pre-trained model

## Overview
["YOLOv3: An Incremental Improvement" ](https://arxiv.org/pdf/1804.02767.pdf) paper can be accessed from [here](https://arxiv.org/pdf/1804.02767.pdf). 

YOLO is used for multiple object detection in a colored image. Version-3 supports detection of 80 different objects. The original model was trained on COCO dataset (for more details refer the paper). Authors have provided the [pre-trained weights](https://pjreddie.com/media/files/yolov3.weights) and the [network architecture information](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg). Here, YOLO_V3 model is generated using the architecture information provided in the "yolov3.cfg" config file. Complete end-to-end network architecture flowchart is provided in the "Model.png" file. For easy understanding of the config file, block numbers have been added and the custom config file is provided in the "cfg" directory of this repository. __Note-__ Block numbers are not to be confused with the layer numbers. Any reference to a layer number is "0" based index number, like in "route" or "shortcut" blocks in the config file.  

<p align="center">
<img src="https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2/blob/master/Result/street.jpg">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2/blob/master/Result/freeway.jpg">
</p>

## Dependencies
* python 3.5.2
* OpenCV 4.1.2
* numpy 1.18.1
* tensorflow 2.2.0

## How to run

Download the pre-trained weights of the YOLO_V3 provided by the authors of the original paper from [here](https://pjreddie.com/media/files/yolov3.weights) and save it in the 'data/' sub-directory. Now run the below command to convert the weights to tensorflow compatible format.

```
git clone https://github.com/varunasthana92/YOLO_v3_implementation_Tensorflow2.git
<download the pre-trained weight file in the "data" sub-directory>
mkdir weights
python3 convert_weights.py
```

Above commands are to be executed only once. The converted weights can then be used with the tensorflow implementation using the below command.
```
python3 detect.py --img_path=data/images/street.jpg
```

__Implementation Notes:__
* Anchor box sizes (provided in the cfg file) are to be normalized with the model size (input layer image size). 

## Contact Information
Name: Varun Asthana  
Email id: varunasthana92@gmail.com

## References
* https://arxiv.org/pdf/1804.02767.pdf
* https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6
* https://github.com/zzh8829/yolov3-tf2
