# YOLO v2

You Only Look Once pytorch implementation for learning and research purposes.
* https://arxiv.org/abs/1612.08242

## Requirements
* Python 3.x
* Numpy
* OpenCV
* Pytorch
* Matplotlib
* PIL

## Usage

Run ```main.py``` for training and testing the implemented YOLO architecture. Set the ```dataset_dir``` variable to the VOC path.
Training and testing parameters may be modified using the ```config.py``` file.
Detection results are saved into text files following the format of any of the following repositories for mAP calculation:
* https://github.com/rafaelpadilla/Object-Detection-Metrics
* https://github.com/Cartucho/mAP

## Results

<img src="https://github.com/cjvargasc/YOLOv2/blob/master/imgs/airplane.png">
<img src="https://github.com/cjvargasc/YOLOv2/blob/master/imgs/boats.png">
<img src="https://github.com/cjvargasc/YOLOv2/blob/master/imgs/motorbike.png">


## acknowledgement
This code is based on the following repositories:
* https://github.com/tztztztztz/yolov2.pytorch
* https://github.com/uvipen/Yolo-v2-pytorch


