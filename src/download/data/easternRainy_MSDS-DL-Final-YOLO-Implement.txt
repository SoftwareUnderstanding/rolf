# MSDS-DL-Final-YOLO-Implement
Author: Eriko Funasato, Sicheng Zhou

This is the final project repo of MSDS 631 Deep Learning Neural Networks course taught by Michael Ruddy at University of San Francisco. In this project, we implement the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf).


## Description of Task
We implement the YOLO detection network in the original paper, and a pre-trained ResNet for comparision on a processed PASCAL VOC 2007 and 2012 [Dataset](https://www.kaggle.com/dataset/734b7b).

Here is the model architecture.

![](images/architecture.png)


## Training Parameters
The original training parameters are:

- Loss Function: multi-part loss function
- Epochs: 135
- Batch Size: 64
- Momentum: 0.9
- Weight Decay: 0.0005
- Learning Rate: 0.01 for 75 epochs, 0.001 for 30 epochs, and 0.0001 for 30 epochs
- Dropout Rate: 0.5

However, our implementation
- Loss Function: multi-part loss function
- Epochs: 79
- Batch Size: 20
- Learning Rate: 0.001

## Results

| Model | mean Average Precision |
|-------|------------------------|
| YOLO Original Paper | 57.9%  |
| YOLO our implementation | 1.537% |
| ResNet34  | 3.018% |

- Here are the metrics curves

| YOLO our implementation | ResNet34 |
|-------------------------|----------|
| ![](images/yolo_on_full_data.jpg) | ![](images/resnet_on_full_data.jpg) |

## Future Improvements
- Learning rate: decay learning rate after every 30 epochs
- Weight Decay: set weight decay parameter in optimizer
- Batches and epochs: larger batch size and more epochs
- Loss: emphasis parameters on class and box
- Data Augmentation
- Pre-train model on ImageNet dataset



## Reference
- https://arxiv.org/pdf/1506.02640.pdf
- https://www.youtube.com/watch?v=n9_XyCGr-MI
- https://github.com/aladdinpersson/Machine-Learning-Collection

