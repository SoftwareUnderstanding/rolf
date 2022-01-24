# keras-yolo2
## Introduction
This repo contains the implementation of YOLOv2 in Keras with Tensorflow backend.

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
**YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi**.

---

## Requirement
- python 3.5
- keras 2.2.4
- tensorflow 1.12.0

## TODO 
- [x] train model
- [ ] predict model
- [ ] mAP Evaluation
- [ ] Tensorborad
   
## train
```bash
cd tools
python train_yolo2.py pascal 'path_to_data'/VOCdevkit/VOC2007
```
## Discussion

## Reference

1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, You Only Look Once: Unified, Real-Time Object Detection, arXiv:1506.02640 (2015).
2. J. Redmon and A. Farhadi, YOLO9000: Better, Faster, Stronger, arXiv:1612.08242 (2016).
3. darkflow, https://github.com/thtrieu/darkflow
4. Darknet.keras, https://github.com/sunshineatnoon/Darknet.keras/
5. YAD2K, https://github.com/allanzelener/YAD2K
6. https://github.com/experiencor/keras-yolo2
