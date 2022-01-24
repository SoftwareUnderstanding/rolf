# ResNet Recurrence

This repository contains codes using TensorFlow to recurrence the ResNet CNN
1. [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
2. [https://arxiv.org/pdf/1603.05027.pdf](https://arxiv.org/pdf/1603.05027.pdf)


datasets: mnist, cifar50


## preparation

1. operation system
only Ubuntu16 support the newest TensorFlow, check version the operation system
```
lsb_release -a
```
update the operation system to Ubuntu 16.04.2 LTS

2. python installation and update
must install python3 to use the newest TensorFlow
```
sudo apt install python3
sudo apt-get install --upgrade python3
```

3. install pip and pip3
first, install setuptools
```
get --no-check-certificate  https://pypi.python.org/packages/source/s/setuptools/setuptools-19.6.tar.gz#md5=c607dd118eae682c44ed146367a17e26
tar -zxvf setuptools-19.6.tar.gz
cd setuptools-19.6
python3 setup.py build
sudo python3 setup.py install
```

second, install pip,pip3
```
wget --no-check-certificate  https://pypi.python.org/packages/source/p/pip/pip-8.0.2.tar.gz#md5=3a73c4188f8dbad6a1e6f6d44d117eeb
tar -zxvf pip-8.0.2.tar.gz
cd pip-8.0.2

python3 setup.py build
sudo python3 setup.py install
```
third, update pip,pip3
```
python3 -m pip install --upgrade pip  
```

4. begin installing newest TensorFlow(1.8.0)
```
pip3 install tensorflow    
```

use the following commands to update python packages

```
pip list --outdated
sudo pip  install  --upgrade SomePackage
```

5. install ROOT
ROOT is a CERN developed software tool used in high energy physics  for
big data processing, statistical analysis, visualisation and storage.
just follow the website to install ROOT:
[https://root.cern.ch/](https://root.cern.ch/)


## dataset introduction

1. MNIST

2. cifar10

3. ImageNet

## using TensorFlow to construct a CNN
the CNN(convolutional neural network) usually contains 3 parts:
1. covolutional layer
2. pooling layer
3. fully connected layer


function of all part:
1. fetch the features of the input images
2. downsample the image

in order to accelerate convergence
ReLU used

```
python3 Plain_Net.py 
```



## using TensorFlow to construct ResNet





```
python3 Plain_Net.py 
```


## Plot and compare to other result
use ROOT for plotting:
```
root Plot.c
```

For the MNIST dataset:
The loss function vs. training epoches, the loss function indicate that the CNN converge after about 200 epoches.
![loss function](https://github.com/horse007666/ResNet/blob/master/Plot/mnist_loss.png)

use 1000 training data to validate
![loss function](https://github.com/horse007666/ResNet/blob/master/Plot/mnist_train_error.png)

use 10000 test data set to validate 
![loss function](https://github.com/horse007666/ResNet/blob/master/Plot/mnist_test_error.png)













