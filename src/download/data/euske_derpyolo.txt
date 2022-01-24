# DerPyOLO

A derpy Python implementation of YOLO (mixture of v1 and v2).

Purely for educational purposes. Useless for production.
(COCO mAP: 0.10)

Original papers:
 - YOLO: https://arxiv.org/abs/1506.02640
 - YOLOv3: https://arxiv.org/abs/1804.02767


## Prerequisites:

 - Python3 (https://www.python.org/)
 - PyTorch (https://pytorch.org/)
 - COCO dataset (https://cocodataset.org/)
 - Pillow (https://python-pillow.org/)
 - OpenCV-Python (https://pypi.org/project/opencv-python/)
 - (optinal)
   - CUDA (https://developer.nvidia.com/cuda-downloads)
   - CuDNN (https://developer.nvidia.com/cudnn)


## Train

 1. Put the following files in `./COCO/` directory:
    - `./COCO/train2017.zip`
    - `./COCO/annotations/instances_train2017.json`

 2. Do this:
    (this can take days.)
```
$ python train.py -n30 -o./yolo_net.pt ./COCO/train2017.zip ./COCO/annotations/instances_train2017.json
```


## Use

### Annotate image:

    $ python detect.py -O. -i./yolo_net.pt image.jpg

### Real-time detection with camera:

    $ python detect.py -V -i./yolo_net.pt

### Webcam demo:

    $ cd demo/
    $ python server.py
    (connect to :4343 via Chrome or Edge)
