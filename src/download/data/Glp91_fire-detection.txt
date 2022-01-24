# fire-detection
Fire detection using InceptionResNet, Python3, Tensorflow


This code performs the smart fire-detection system core.
This system implements a neural network to detect flames in images or videos.

The net used: ResNet50 witch is a resdual Neural Network for Deep Residual Learning for Image Recognition
link to paper:https://arxiv.org/abs/1512.03385

dependencies: Python 3,6
              openCV, 
              TensorFlow.

model path:'path/fire-resnet-graph/', there are no checkpoints, use your own models.

training configuration file path: object_detection/path/fire-resnet-graph/pipeline.config


to run the fire detector:
python3 fire-detection.py
