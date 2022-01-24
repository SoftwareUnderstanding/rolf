# mobilenetv1_keras

JUST FOR FUN!
I had read an article :MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (https://arxiv.org/abs/1704.04861).<br>

Try to use Keras to train mobilenetv1 and compare using mobilenetv1 architecture and without using mobilenetv1 architecture.<br>
Although the mobilenetv1 architecture simplify the VGG16, there are some difference between with it.
I code it using Keras according to article and classify successfully using the Kaggle dataset:https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data <br>

I use 4000 cats.jpg and 4000 dogs.jpg for training and use 3000 cats.jpg and 3000 dogs.jpg for testing. The index as follow:<br>

>---train_cat_dog<br>
>>>--cat<br>
>>>--dog<br>

> ---validation_cat_dog<br>
>>>--cat<br>
>>>---dog<br>
    
The mobilenetv1 summary as follow:<br>

![image](https://github.com/zhucheng725/mobilenetv1_keras/blob/master/mobilenetv1_summary.png)<br>

And the classification as follow:

![image](https://github.com/zhucheng725/mobilenetv1_keras/blob/master/mobilenetv1.png)<br>

We can see how easy to train the network using few parameters.

Contrasty, if I change depthwise seperable convolution to fully convolution, it cost lots of parameters and the final result as follow:

![image](https://github.com/zhucheng725/mobilenetv1_keras/blob/master/summary.png)<br>


![image](https://github.com/zhucheng725/mobilenetv1_keras/blob/master/training.png)<br>

Because my PC own the GTX1060 3G and can not easy to train the vgg16, I simplify the network and get an 50% accuracy. That means we can not use this vgg16 to train 2 classes. You can train the easy network such as Cifar10 to test.<br>
