# MEME Tracking Algorithm 
Our algorithm has 2 Deep Learning Models.  
- Object Tracking algorithm
- Keyword Spotting algorithm


## 1. Object Tracking: Yolov5 + Deep Sort with PyTorch

### Description

The implementation is based on two papers & Github Repository:
- Object Tracking(https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf

### Requirements

Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:

`pip install -U -r requirements.txt`

### Tracking

Tracking can be run on most video formats

```bash
python3 track.py --source ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

MOT compliant results can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```

## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.

## 2. Keyword Spotting
