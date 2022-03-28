# SSD Flask Docker

A SSD object detection web project implemented with Flask and Docker.

The referenced paper is: SSD: Single Shot MultiBox Detector https://arxiv.org/pdf/1512.02325.pdf


## Main Requirements and Versions

```
python >= 3.5
pytorch >= 1.0.0
cuda 9.0
cudnn 7
```



&nbsp;

## Quick Start

### 1. Run web service in Docker Container
```
docker pull wunaiq/ssd_flask_docker:py36_pytorch1.0.0_cu90
docker run -itd \
           --name=server \
           --network=host \
           -p 8008:8008 \
           wunaiq/ssd_flask_docker:py36_pytorch1.0.0_cu90
```
Now, the web server is runing on: [http://0.0.0.0:8008](http://0.0.0.0:8008)

### 2. Test in Docker Contanier
```
docker run -it \
           --name=client \
           --network=host \
           wunaiq/ssd_flask_docker:py36_pytorch1.0.0_cu90 \
           /bin/bash

# Run test scrip in contanier to ensure that the environment meets the requirements.
# There are some test images in container for simple testing, more detatils about testing are in Web Test & Eval.

python web_test.py \
       -restype=bboxes \
       -data_root=./custom_data/ \
       -save_dir=./custom_results \
       -test_url=http://0.0.0.0:8008/test \
       -cuda=True                                    
```


&nbsp;

---
## Web Test & Eval

1. Test on custom data

```
cd ./web_test
mkdir custom_data  # and put your test images in this folder
python web_test.py -restype=image \
                   -data_root=./custom_data \
                   -save_dir=./custom_results \
                   -cuda=True
```
-restype: 
* image: test on custom data and return image with bboxes drew
* bboxes: test on custom data and return locations of bboxes
* precision: eval on VOC and return mAP

-save_dir: directory of all the web test results
-cuda: test on GPU or not

2. Eval on VOC 
```
python web_test.py -restype=precision \
                   -data_root=../app/SSDdetector/data/VOCdevkit \
                   -save_dir=./voc_results \
                   -cuda=True
```
&nbsp;

---
## SSD Training

### 1. Preparation

Download COCO:

```
cd ./SSDdetector
./data/scripts/COCO2014.sh
```

Download VOC:

```
./data/scripts/VOC2007.sh
./data/scripts/VOC2012.sh
```

Download the fc-reduced VGG-16 PyTorch base network weights:
```
mkdir weights
cd ./weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

```


### 2. Trian 

```
cd ./app/SSDdetector/
python3 ./trainval/train.py
```

*Training loss visualization with visdom:
```
pip3 install visdom
python3 -m visdom.server
# Then navigate to http://localhost:8097/ during training
```

## SSD Eval

```
cd ./app/SSDdetector/
python3 ./trainval/eval.py
```

Performance on VOC2007 Test：

| mAP | FPS(GTX Titan X)| FPS(CPU)|
|:-:|:-:|:-:|
| 77.49 % |22.62|2.28|


&nbsp;

&nbsp;

&nbsp;


---

### Reference:

1. Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//European conference on computer vision. Springer, Cham, 2016: 21-37.
2. https://github.com/amdegroot/ssd.pytorch.git
3. https://github.com/jomalsan/pytorch-mask-rcnn-flask.git
4. https://github.com/imadelh/ML-web-app.git
