# DynamicRCNN-mmdet

Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training

paper address: https://arxiv.org/pdf/2004.06002.pdf 

## Requirements

- Linux OS
- Python 3.7 (Python 2 is not supported)
- PyTorch 1.2.0
- torchvision 0.4.0 
- mmdetection tag: v1.2.0
- mmcv 0.4.3
- CUDA 10.0
- GCC(G++) 5.4.0 or higher


## Installation

a. Create a conda virtual environment and activate it (Optional but recommended).

```shell
conda create --name dynamic python=3.7
conda activate dynamic
```

b. Install pytorch and torchvision.  
pip is recommended, 
```shell
pip install torch==1.2.0 torchvision==0.4.0  
```


c. Install mmdet (other dependencies wil be installed automatically).

```shell
pip install cython
pip install easydict
pip install -r requirements.txt
pip install -v -e .
```


d. Prepare dataset and checkpoint file.

Download [coco dataset](http://cocodataset.org/#download) and [checkpoint file](https://download.pytorch.org/models/resnet50-19c8e357.pth)

Fold structure should be as follows:

```
DynamicRCNN-mmdet
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── backbone
│   ├── resnet50-19c8e357.pth
```


## Features
- Synchronize on multi-gpus


## Train
```shell
bash scripts/train.sh
```

## Test
```shell
bash scripts/test.sh
```

## Results

### Faster-RCNN

