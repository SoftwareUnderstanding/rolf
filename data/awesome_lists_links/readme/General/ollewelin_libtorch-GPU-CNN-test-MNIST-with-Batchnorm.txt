# libtorch-GPU-CNN-test-MNIST-with-Batchnorm
Test add batchnorm layers.

Was modifyed this code with adding batchnorm layer between each convolution layers

https://github.com/goldsborough/examples/tree/cpp/cpp/mnist

### Youtube video
https://www.youtube.com/watch?v=wLQbXEORgFA

### MNIST datasets

#### MNIST Fashion dataset

https://github.com/zalandoresearch/fashion-mnist

#### Or use the old MNIST digits dataset

http://yann.lecun.com/exdb/mnist/

### Without batch norm layers connected (on MNIST digits dataset)

    Train Epoch: 10 [59584/60000] Loss: 0.0165
    Test set: Average loss: 0.0429 | Accuracy: 0.987

### With batch norm layer attached (on MNIST digits dataset)

    Train Epoch: 10 [59584/60000] Loss: 0.0120
    Test set: Average loss: 0.0315 | Accuracy: 0.989



Example print out

    Train Epoch: 10 [59584/60000] Loss: 0.0120
    Test set: Average loss: 0.0315 | Accuracy: 0.989
    Print Model weights parts of conv1 weights kernels
    0.0714 -0.0887 -0.2127 -0.1545 -0.0813
    0.1184  0.1395  0.0606  0.0129  0.0564
    -0.0033  0.1634  0.2492  0.1134  0.0322
    -0.0914 -0.0334  0.0359  0.1716  0.1377
    -0.1568 -0.1173 -0.1753 -0.1878 -0.0052
    [ CUDAFloatType{5,5} ]

## To be continue.... code under development..
### Continue exploring Libtorch C++ with OpenCV towards a plane simple ResNet-34 training from scrach with custom image dataset.

The code snippet :

        under construction main.cpp
        develop backup main (copy).cpp
        ..

I will try to do a (mid level programming) of a fix plain ResNet-34 (hardcoded ResNet-34 not generic ResNet-X with bottlenecks etc).
Toghether with custom data set using OpenCV for a classification of color images or video stream. Not need using torchvision for this yet.

#### Prepare dataset tensor from abriarity size of test.jpg input image

The read_data() function adapt the test.jpg to a tensor with shape 

       Tensor [1, 3, 224, 224]
        
To fit as input for the future ResNet-34 classification model

![](Prepare_1_3_224_224_tensor_from_test_jpg.png)

#### Flowers datasets from kaggle

https://www.kaggle.com/alxmamaev/flowers-recognition

#### Resnet paper

Paper :
Figure 3, resnet-34

https://arxiv.org/pdf/1512.03385.pdf

#### Excellent explanation of ResNet in general

https://erikgaas.medium.com/resnet-torchvision-bottlenecks-and-layers-not-as-they-seem-145620f93096#_=_

#### Disscusion regarding dataloader

https://discuss.pytorch.org/t/libtorch-how-to-use-torch-datasets-for-custom-dataset/34221/2

https://krshrimali.github.io/Training-Network-Using-Custom-Dataset-PyTorch-CPP/

https://krshrimali.github.io/Custom-Data-Loading-Using-PyTorch-CPP-API/

#### Disscusion regarding OpenCV -> Tensor and Tensor -> OpenCV

https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/5

### 5 classes test 5 diffrent flowers (4-conv layer network)

![](5_classes_flowers.png)

        main 5_classes.cpp
        
        file_names_5_classes_500_jpg_img.csv

#### other links

https://discuss.pytorch.org/t/libtorch-how-to-save-model-in-mnist-cpp-example/34234
   
   
