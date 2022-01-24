# Wheat Head Detection using PyTorch

This is a PyTorch implementation of FasterRCNN using Resnet50 backbone. We will also leverage the pretrained model available in PyTorch. Additional reading: https://arxiv.org/abs/1506.01497

I will use the below in this project:
* Pretrained FasterRCNN from torchvision,
* Use of Albumentations
* Calculate Validation IOU
* Early Stopping
* Ensemble boxes using weighted box fusion
* Generate and view results on public test data set

![](https://github.com/harinath0906/Object-Detection-using-PyTorch/raw/master/FasterRCNN.png)

## Test Results:
![Test Results](https://github.com/harinath0906/Object-Detection-using-PyTorch/raw/master/Test_Results_Screenshot.png)
