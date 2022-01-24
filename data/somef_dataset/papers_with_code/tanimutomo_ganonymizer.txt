# GANonymizer
*GANonymizer Image Annonymization Method Using Object Detection and Generative Adversarial Networks*
Improved version, GANonymizerV2 is available [here](https://github.com/tanimutomo/ganonymizerv2).

## Note
The code of this repository is leverated on the following repositories.
Also we use the pre-trained models opend publicly in their repository.
1. https://github.com/weiliu89/caffe/tree/ssd 
2. https://github.com/satoshiiizuka/siggraph2017_inpainting
3. https://github.com/ayooshkathuria/pytorch-yolo-v3

## Requirements
Written in the `docker/gpu_requirements.txt`.

## Preparation
1. git clone
2. download cfgs and weights for (SSD512 or YOLOV3) and GLCIC
```
cd ganonymizer
```
### SSD cfgs
```
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/cfgs/deploy.prototxt -P ganonymizer/src/detection/ssd/cfgs 
```
### SSD weights
```
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel -P ganonymizer/src/detection/ssd/weights
```
### YOLO-V3 (this link is the author page of YOLOV3.)
```
wget https://pjreddie.com/media/files/yolov3.weights -P ganonymizer/src/detection/yolov3/weights
```
### GLCIC weights
```
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/glcic/weights/completionnet_places2.pth -P ganonymizer/src/inpaint/glcic/weights
```
In terms of SSD, We use [Weiliu's SSD model](https://github.com/weiliu89/caffe/tree/ssd). You can download SSD's cfgs and weights from [this](https://github.com/weiliu89/caffe/tree/ssd).  
GLCIC model we use is [Iizuka's model](https://github.com/satoshiiizuka/siggraph2017_inpainting). We convert the model's type from torch7 to pytorch.


## Usage (Docker)
1. Install Docker and Docker-compose
1. custom config in ganonymizer/main.py
1. Build the container
`docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml build {/--gpu}`
1. Execute GANonymizer to the input image specified in the main.py
`docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml run experiment python3 main.py`


## Reference
The followings are the main reference paper.
1. https://dl.acm.org/citation.cfm?id=3073659
2. https://link.springer.com/chapter/10.1007%2F978-3-319-46448-0_2
3. https://arxiv.org/abs/1804.02767


## Details of this paper
Title: GANonymizer: Image Anonymization Method Integrating Object detection and Generative Adversarial Networks  
Authors: Tomoki Tanimura, Makoto Kawano, Takuro Yonezawa, Jin Nakazawa
