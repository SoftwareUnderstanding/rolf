# CIFAR-10 TensorFlow ResNet

This program performs image classification on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It uses a ResNet with identity mappings, similar to the one described by Kaiming He et. al. in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) and [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf). In order to facilitate automated tuning and experimentation, all settings and hyperparameters are defined in params.py. 

## Getting Started

1. Download the python version of the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Extract cifar-10-batches-py
3. Edit params.py so that DATA_DIR points to the directory containing cifar-10-batches-py. 
4. Run train.py. 

### Prerequisites

Python 3.6, TensorFlow 1.5, and NumPy should be installed before running this program. 

## Author

* **Sean Soleyman** - [seansoleyman.com](http://seansoleyman.com)

## License

MIT License

Copyright (c) 2018 Sean Soleyman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## References
1. Aurelien Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
2. [Kaiming He et. al., Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
3. [Kaiming He et. al., Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)
4. [Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

