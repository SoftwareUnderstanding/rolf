## SSD detector

This project implements object detection algorithm described in https://arxiv.org/pdf/1512.02325.pdf. Source paper added to docs folder.
Python implementation uses PyTorch framework and OpenCV library.
Model training and evaluation have been tested under Ubuntu 18.04 using OpenCV 4.1.1, PyTorch 1.3.1, Tensorboard 2.1.0

Evaluation example for model with pretrained backbone (MobileNet-v2) and trained (on Pascal VOC-2012 dataset) classification and regression heads

![Alt text](attachments/ssd_demo.gif?raw=true "Model's evaluation without NMS")


### Training

Use train.py script to start model's training. Arguments for script are described in parse_cmd_args() function.
SSD with MobileNet-v2 backbone has been trained on Pascal VOC 2012 dataset. With fixed backbone's weights (only heads were training) model has achieved 0.4 mAP on the validation dataset.


### Testing

Use deploy.py script to evaluate model with webcam/videofile stream. Pretrained model has been attached to the repo.
Example of proper launch:

python3 deploy.py --model=Ep87Btch133_SSD_Mobilenetv2_6fm6p21c_2020_02_14_17_51_08.torchmodel --threshold=0.1



