# Introduction
## Models:
- [YOLOv1](https://arxiv.org/abs/1506.02640)
- [YOLOv2](https://arxiv.org/abs/1612.08242)
- [YOLOv3](https://arxiv.org/abs/1804.02767)

## Backbone:
- DarkNet
- EfficientNet

## Custom Data Enchancement method:
- Normalize
- Resize
- RandomHorizontalFlip
- RandomVerticalFlip

# Code structure description
```
yolo.ai
├── cfg                 # The directory where model config file is located (darknet, efficientnet, etc)
├── tests               # Implmentation test cases
├── yolo                # YOLO implementation code bases
│   ├── data            # Base data directory
│   │   ├── datasets    # Contain datasets such as pascal-voc
│   │   └── transforms  # Custom transforms for Dataset
│   ├── models          # Base model directory
│   │   ├── arch        # YOLO model assembly place
│   │   ├── backbones   # All backbone network gathering here
│   │   ├── detectors   # Assembly of all types of detectors
│   │   ├── losses      # The gathering place of all loss functions
│   │   ├── metrics     # Metrics functions for bounding boxes and losses
│   │   └── modules     # Individuals modules for network building
│   └── utils           # Utilites file for visualization and network
```
# Installation

# References
- [Dive really deep into yolov3 a beginner guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)

- [@aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection)