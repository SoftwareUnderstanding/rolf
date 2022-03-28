## Introduction

This is the system that does the object segmentation on frames/images. 

## Source

The mrcnn folder is based on https://github.com/matterport/Mask_RCNN and converted to latest tensorflow version (2.1.0).

## What does it do?

### Download the pretrained model

The author provided pretraind model to download https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
We can also train on custom dataset with this repo and get a new model.

### Tensorflow implementation for Mask-RCNN

Run detection to get object segmentations based on the algorithm in this paper: https://arxiv.org/pdf/1703.06870.pdf


## Run locally
Add sample.jpg you want to run to maskrcnn/images folder
```
pip3 install -r requirements.txt
python3 instance_segmentation.py
```
Return results in following format: 
```buildoutcfg
[{'class_name': 'person', 'offset': (0.29248046875, 0.22509765625), 'size_percentage': 0.17861270904541016},  {'class_name': 'chair', 'offset': (0.59912109375, 0.083984375), 'size_percentage': 0.03147125244140625}]

```
class_name is the object name detected in the image
offset is the location (center point) of where the object appears in the image
size_percentage is the size of the object that takes up how much percentage of the entire image 