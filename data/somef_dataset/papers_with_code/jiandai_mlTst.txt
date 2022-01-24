## Key Ingredients for Deep Learning 


- Hardware at the core: Nvidia GPU
  - Kepler eg K80
  - Maxwell eg M60
  - Pascal ([wiki](https://en.wikipedia.org/wiki/Pascal_(microarchitecture))) eg P100
  - Volta (to come)

- System

- Computing model: [CUDA](https://developer.nvidia.com/cuda-downloads) ([wiki](https://en.wikipedia.org/wiki/CUDA))
  - CUDA 7
  - CUDA 8

- Deep Learning frameworks (name, backed-by)

  - tensorflow, Google
  - keras, Google
  - caffe/caffe2, Facebook
  - torch/pytorch, Facebook
  - mxnet, Amazon for AWS
  - cntk, Microsoft
  - Theano,
  - [dl4j](https://deeplearning4j.org/),


- Model architecture (link to [keras application](https://keras.io/applications/) if there is one)

  - LeNet
  - AlexNet
  - [VGG16](https://keras.io/applications/#vgg16), [VGG19](https://keras.io/applications/#vgg19)
  - GoogLeNet
  - [Inception](https://keras.io/applications/#inceptionv3)
  - [Xception](https://keras.io/applications/#xception)
  - [ResNet](https://keras.io/applications/#resnet50)

- Special purposed architecture

  - U-net ([arxiv](https://arxiv.org/abs/1505.04597))
  - V-net
  - E-net

- Building blocks
  - densely connected layer ([my tutorial: ann101](https://github.com/jiandai/mlTst/blob/master/tensorflow/ann101.ipynb))
  - locally connected layer
  - convolutional layer ([my tutorial: cnn from scratch](https://github.com/jiandai/mlTst/blob/master/semeion.ipynb), [my tutorial: minimal cnn for MNIST](https://github.com/jiandai/mnist/blob/master/kaggle-mnist-tst-0-1.ipynb))
  - pooling layer
  - dropout layer

- Techniques
  - Optimization (a lot)  
  - Batch normalization
  - Data augmentation ([my tutorial](https://github.com/jiandai/mlTst/blob/master/keras/image-data-augmentation-by-keras.ipynb), [my tutorial using MNIST](https://github.com/jiandai/mnist/blob/master/image-data-augmentation-by-keras.ipynb))
  - Feature extraction using pretained weight ([my tutorial: VGG and Xception](https://github.com/jiandai/mlTst/blob/master/keras/DL-features.ipynb))
  - Transfer learning ([my tutorial: predict dogs-vs-cats using ImageNet-trained Xception net](https://github.com/jiandai/dogs-vs-cats/blob/master/dogs-vs-cats-ex.ipynb))
  - Visualization computational graph and training process ([my tutorial: visualize computational graph](https://github.com/jiandai/mlTst/blob/master/tensorflow/tensorboard-101.ipynb), [my tutorial: training and validation loss monitoring](https://github.com/jiandai/mlTst/blob/master/tensorflow/mnist-with-2-conv-layer-vis-in-tensorboard.ipynb))
