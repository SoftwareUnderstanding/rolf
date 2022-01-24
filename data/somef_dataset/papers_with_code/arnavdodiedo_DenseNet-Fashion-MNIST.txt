# DenseNet-Fashion-MNIST

An implementation of DenseNet trained on the Fashion MNIST dataset.

#### The following commands download the dataset:
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

Or get the dataset at: https://www.kaggle.com/zalando-research/fashionmnist/version/4#_=_


#### References:
1. Official DenseNet Keras implementation - https://github.com/cmasch/densenet
2. DenseNet Paper - https://arxiv.org/pdf/1608.06993.pdf
