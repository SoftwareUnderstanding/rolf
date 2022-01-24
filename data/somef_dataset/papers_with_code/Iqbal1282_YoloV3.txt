# YoloV3
Object detection with yolov3 algorithm using  Tensorflow-2 

Clone or Download the files.

1. Downlaod the coco  weight from https://pjreddie.com/media/files/yolov3.weights  and keep in the weights directory 


### Prerequisites
This project is written in Python 3.7 using Tensorflow 2.0 (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision) and absl (flags) packages.



#### Installation

```bash
pip3 install -r ./requirements.txt
wget https://pjreddie.com/media/files/yolov3.weights -O ./yolov3.weights
```


#### YouTube Tutorial
'''
  https://youtu.be/Rb2mcbyhgLI
'''



#### Detect

```bash
python detect.py 
```

#### Use

0pen the detect.py file check the name of directory of images and video files . 
You can use your image and keep it in the data folder. And change the name of the directory 
according to image name or video file name  in the line 11 and 15. 

U can change the input type of the main function in the line 86 to 'image', 'video'
,and 'camera' to use different types of file. If camera is chosen your webcam will 
open. 
```

#### Papers and thanks

- [YOLO website](https://pjreddie.com/darknet/yolo/)
- [YOLOv3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [NMS paper](https://arxiv.org/pdf/1704.04503.pdf)
- [NMS implementation](https://github.com/bharatsingh430/soft-nms)
- [GIOU Paper](https://giou.stanford.edu/GIoU.pdf)
- [DarkNet Implementation](https://github.com/pjreddie/darknet)
- [YOLO implementation](https://github.com/zzh8829/yolov3-tf2)


[image]: ./YOLO_V3/output_0.jpg "Logo Title Text 2"


#### output_0.jpg

#### Sample:

![output Result](https://github.com/Iqbal1282/YoloV3/blob/master/YOLO_V3/output_0.jpg)

