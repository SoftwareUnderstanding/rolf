# Detection
This a Faster R-CNN model based on paper << Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.>>
http://arxiv.org/pdf/1506.01497.pdf written for learning purposes
The program structure is base on Tensorflow's 'nmt'https://github.com/tensorflow/nmt and endernewton's workship tf-faster-rcnn
https://github.com/endernewton/tf-faster-rcnn
PS:For a good and more up-to-date implementation for faster/mask RCNN with multi-gpu support, please see endernewton's new example
https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

# DataSet
MCOCO2014

# Pre-trained model
download location:https://github.com/tensorflow/models/tree/master/research/slim
VGG_16
ResNetV1_50
Inception_V3 unable to work temporarily,cause Tensorflow's avg pooling method don't supoort dynamic ksize and stride temporarily,
see Issue https://github.com/tensorflow/tensorflow/issues/26961

# Environment
old:
tensorflow-gpu: 1.13.1
CUDA: 10.0.130
CUDNN: 7.3.1
GPU: Nvida RTX-2070
new:
tensorflow-gpu: 1.9.0
CUDA: 9.0.176
CUDNN: 7.3.0
GPU: Nvida RTX-2070
# This model is unable to work now,cause (fixed)
UnknownError (see above for traceback): Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
	 [[node vgg_16/conv1/conv1_1/Conv2D (defined at F:\Pycharm\PyCharm Community Edition 2018.3.5\workspace\Detection\models\vgg_16.py:20) ]]
It looks like the error was caused by unmatching between CUDA and CUDNN, but i am not 100% sure.
Solution
Change environment to the new one may fix this problem
# This model is unable to work now, cause the calculation from model is wrong when i draw calculated bounding box on images, remain fixing and updating

