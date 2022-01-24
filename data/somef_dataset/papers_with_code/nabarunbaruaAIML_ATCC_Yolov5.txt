# ATCC  :  YOLOv5+Deep Sort with PyTorch+ Easy OCR(*) + ESRGAN

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FnabarunbaruaAIML%2FATCC_Yolov5&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


![](atcc_new.gif)

## Introduction

This repository contains a moded version of PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)
It filters out every detection that is not a Number Plate. The detections of Vehicle Number Plates are then passed to a Deep Sort algorithm (https://github.com/ZQPei/deep_sort_pytorch) which tracks the same along with Pytorch 1.7.
The main reason to only detect Number plates is that the deep association metric is trained on a Vehicle Number Plate ONLY dataset.The detections are then cropped and subjected to Super Resolution Technique ESRGAN ( Training of the ESRGAN to get better Resolution Number Plate is done) to get high resolution number plates followed by application of EasyOCR on the same images.The registration number plates after being read are then logged into a CSV  .

## TODO

Going forward , the crux of the solution is to detect track and identify the vehicles which have crossed the speed thresholds of the area. In the event of such an occurence, a Chalan is expected to be shot to the holder of the registration.
Multi-Camera feed is also in the road map with Amazon Textract at the OCR department and also an integration with vahan.com for chalan propagation. 


## Description

The implementation is based on the following:
- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf
-YOLOv5
https://github.com/ultralytics/yolov5
- ESRGAN
https://arxiv.org/abs/1809.00219

## Requirements

Python 3.8 or later with all environment.yml dependencies installed, including torch>=1.7. To install run:

`conda env create -f environment.yml`

All dependencies are included in the environment.yml file. 


## Pre-requisites before running the Driver Script (Track.py)

1. Clone the repository recursively:
`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`
If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Github block pushes of files larger than 100 MB (https://help.github.com/en/github/managing-large-files/conditions-for-large-files). Hence you need to download two different weights: the ones for yolov5 and the ones for deep sort.

- download the yolov5 weight from the latest release https://drive.google.com/file/d/1vobC7lH0e7f3H-39I-N1IB1qJ5cfAcCR/view?usp=sharing Place the downlaoded `.pt` file under `yolov5/weights/`
- download the deep sort weights from https://drive.google.com/file/d/1ZdkGDR4V4OzREl-QSNJZD_K9n_XDStD1/view?usp=sharing Place ckpt.t7 file under`deep_sort/deep/checkpoint/`


## Running 

Running can be run on most video formats (or RTSP Camera feeds)

```bash
python3 track.py --source ...
```

- Video:  `--source fileName.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

Note: However, some default values set.
Like  TestVideo.mp4 is what is used for the demonstration.

Multi Object compliant results can be saved to `inference/output` by 
```bash
python3 track.py --source ... --save-txt
```


## Other information
For further details, please refer the research papers.

