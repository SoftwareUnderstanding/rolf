# ConvolutionNN
> This is a repository of some famous convolution neural network model. 
> Just used to note and review. It's my plesure if you get something from here.

## AlexNet
* Reference
    * Krizhevsky, A., Sutskever, I., & Hinton, G. (2014). ImageNet Classification with Deep Convolutional Neural. In Neural Information Processing Systems (pp. 1-9). [> Detail](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* Require
    * tensorflow >= 1.5
    * matplotlib
* Note
    * debug
* Experimental Result
    * not finished

## VGGNet
* Reference
    * Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. [> Detail](https://arxiv.org/abs/1409.1556)
* Require
    * tensoflow >= 1.5
    * matplotlib
    * numpy
* Note
    * For VGG19,VGG16C,VGG16D model, batch_normalization is added.
    * This code not implement the function of save model ,but summarys will be saved. Since this models are trained using small dataset such as cifar10 and mnist and just try to validate the performance of models.
* Experimental Result
    > VGGNet19<br>
    ![image](result/VGGNet19.png)

    > VGGNet16D<br>
    ![image](result/VGGNet16D.png)

    > VGGNet16C<br>
    ![image](result/VGGNet16C.png)


