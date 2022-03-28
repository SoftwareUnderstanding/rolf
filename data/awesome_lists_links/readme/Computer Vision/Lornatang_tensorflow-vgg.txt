# 论文地址：[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

# 代码地址：[https://https://github.com/Lornatang/tensorflow-vgg](https://github.com/Lornatang/tensorflow-vgg)

VGGNet是牛津大学计算机视觉组（VisualGeometry Group）和GoogleDeepMind公司的研究员一起研发的的深度卷积神经网络。VGGNet探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3*3的小型卷积核和2*2的最大池化层，VGGNet成功地构筑了16~19层深的卷积神经网络。VGGNet相比之前state-of-the-art的网络结构，错误率大幅下降，并取得了ILSVRC 2014比赛分类项目的第2名和定位项目的第1名。同时VGGNet的拓展性很强，迁移到其他图片数据上的泛化性非常好。VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3*3）和最大池化尺寸（2*2）。到目前为止，VGGNet依然经常被用来提取图像特征。VGGNet训练后的模型参数在其官方网站上开源了，可用来在特定的图像分类任务上进行再训练（相当于提供了非常好的初始化权重），因此被用在了很多地方。

![图1 VGGNet各级别网络结构图](http://upload-images.jianshu.io/upload_images/11059787-a1d7745b966e0c1b?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![图2 VGGNet各级别网络参数量](http://upload-images.jianshu.io/upload_images/11059787-f7b840fa2203c55a?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

VGGNet论文中全部使用了3*3的卷积核和2*2的池化核，通过不断加深网络结构来提升性能。图1所示为VGGNet各级别的网络结构图，图2所示为每一级别的参数量，从11层的网络一直到19层的网络都有详尽的性能测试。虽然从A到E每一级网络逐渐变深，但是网络的参数量并没有增长很多，这是因为参数量主要都消耗在最后3个全连接层。前面的卷积部分虽然很深，但是消耗的参数量不大，不过训练比较耗时的部分依然是卷积，因其计算量比较大。这其中的D、E也就是我们常说的VGGNet-16和VGGNet-19。C很有意思，相比B多了几个1*1的卷积层，1*1卷积的意义主要在于线性变换，而输入通道数和输出通道数不变，没有发生降维。

训练时，输入是大小为224*224的RGB图像，预处理只有在训练集中的每个像素上减去RGB的均值。

VGGNet拥有5段卷积，每一段内有2~3个卷积层，同时每段尾部会连接一个最大池化层用来缩小图片尺寸。每段内的卷积核数量一样，越靠后的段的卷积核数量越多：64-128-256-512-512。其中经常出现多个完全一样的3*3的卷积层堆叠在一起的情况，这其实是非常有用的设计。如图3所示，两个3*3的卷积层串联相当于1个5*5的卷积层，即一个像素会跟周围5*5的像素产生关联，可以说感受野大小为5*5。而3个3*3的卷积层串联的效果则相当于1个7*7的卷积层。除此之外，3个串联的3*3的卷积层，拥有比1个7*7的卷积层更少的参数量，只有后者的(3*3*3)/(7*7)=55%。最重要的是，3个3*3的卷积层拥有比1个7*7的卷积层更多的非线性变换（前者可以使用三次ReLU激活函数，而后者只有一次），使得CNN对特征的学习能力更强。

![图3  两个串联3´3的卷积层功能类似于一个5´5的卷积层](http://upload-images.jianshu.io/upload_images/11059787-f6b506c5e15b34a4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

VGGNet在训练时有一个小技巧，先训练级别A的简单网络，再复用A网络的权重来初始化后面的几个复杂模型，这样训练收敛的速度更快。在预测时，VGG采用Multi-Scale的方法，将图像scale到一个尺寸Q，并将图片输入卷积网络计算。然后在最后一个卷积层使用滑窗的方式进行分类预测，将不同窗口的分类结果平均，再将不同尺寸Q的结果平均得到最后结果，这样可提高图片数据的利用率并提升预测准确率。在训练中，VGGNet还使用了Multi-Scale的方法做数据增强，将原始图像缩放到不同尺寸S，然后再随机裁切224´224的图片，这样能增加很多数据量，对于防止模型过拟合有很不错的效果。实践中，作者令S在[256,512]这个区间内取值，使用Multi-Scale获得多个版本的数据，并将多个版本的数据合在一起进行训练。图4所示为VGGNet使用Multi-Scale训练时得到的结果，可以看到D和E都可以达到7.5%的错误率。最终提交到ILSVRC 2014的版本是仅使用Single-Scale的6个不同等级的网络与Multi-Scale的D网络的融合，达到了7.3%的错误率。不过比赛结束后作者发现只融合Multi-Scale的D和E可以达到更好的效果，错误率达到7.0%，再使用其他优化策略最终错误率可达到6.8%左右，非常接近同年的冠军Google Inceptin Net。同时，作者在对比各级网络时总结出了以下几个观点：（1）LRN层作用不大（VGGNet不使用局部响应标准化(LRN)，这种标准化并不能在ILSVRC数据集上提升性能，却导致更多的内存消耗和计算时间。）；（2）越深的网络效果越好；（3）1*1的卷积也是很有效的，但是没有3*3的卷积好，大一些的卷积核可以学习更大的空间特征。

![图4  各级别VGGNet在使用Multi-Scale训练时的top-5错误率](http://upload-images.jianshu.io/upload_images/11059787-1f73584bcee6d753?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在训练的过程中，比AlexNet收敛的要快一些，原因为：（1）使用小卷积核和更深的网络进行的正则化；（2）在特定的层使用了预训练得到的数据进行参数的初始化。

对于较浅的网络，如网络A，可以直接使用随机数进行随机初始化，而对于比较深的网络，则使用前面已经训练好的较浅的网络中的参数值对其前几层的卷积层和最后的全连接层进行初始化。

## 现在我们来使用TensorFlow实现VGG19

下面算一下每一层的像素值计算： 
输入：224*224*3 
1. conv3 - 64（卷积核的数量）：kernel size:3 stride:1 pad:1 
像素：（224-3+2*1）/1+1=224 224*224*64 
参数： （3*3*3）*64 =1728 

2. conv3 - 64：kernel size:3 stride:1 pad:1 
像素： （224-3+1*2）/1+1=224 224*224*64 
参数： （3*3*64）*64 =36864 

3. pool2 kernel size:2 stride:2 pad:0 
像素： （224-2）/2 = 112 112*112*64 
参数： 0 

4.conv3-128:kernel size:3 stride:1 pad:1 
像素： （112-3+2*1）/1+1 = 112 112*112*128 
参数： （3*3*64）*128 =73728 

5.conv3-128:kernel size:3 stride:1 pad:1 
像素： （112-3+2*1）/1+1 = 112 112*112*128 
参数： （3*3*128）*128 =147456 

6.pool2: kernel size:2 stride:2 pad:0 
像素： （112-2）/2+1=56 56*56*128 
参数：0 

7.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*128）*256=294912 

8.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*256）*256=589824 

9.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*256）*256=589824 

10.pool2: kernel size:2 stride:2 pad:0 
像素：（56 - 2）/2+1=28 28*28*256 
参数：0 

11. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*256）*512 = 1179648 

12. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*512）*512 = 2359296 

13. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*512）*512 = 2359296 

14.pool2: kernel size:2 stride:2 pad:0 
像素：（28-2）/2+1=14 14*14*512 
参数： 0 

15. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 

16. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 

17. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 

18.pool2:kernel size:2 stride:2 pad:0 
像素：（14-2）/2+1=7 7*7*512 
参数：0 

19.FC: 4096 neurons 
像素：1*1*4096 
参数：7*7*512*4096 = 102760448 

20.FC: 4096 neurons 
像素：1*1*4096 
参数：4096*4096 = 16777216 

21.FC：1000 neurons 
像素：1*1*1000 
参数：4096*1000=4096000

总共参数数量大约138M左右。
本文主要工作计算了一下VGG网络各层的输出像素以及所需参数，作为一个理解CNN的练习，VGG网络的特点是利用小的尺寸核代替大的卷积核，然后把网络做深，举个例子，VGG把alexnet最开始的一个7*7的卷积核用3个3*3的卷积核代替，其感受野是一样。关于感受野的计算可以参照另一篇博文。 
AlexNet最开始的7*7的卷积核的感受野是：7*7 
VGG第一个卷积核的感受野：3*3 
第二个卷积核的感受野：（3-1）*1+3=5 
第三个卷积核的感受野：（5-1）*1+3=7 
可见三个3*3卷积核和一个7*7卷积核的感受野是一样的，但是3*3卷积核可以把网络做的更深。VGGNet不好的一点是它耗费更多计算资源，并且使用了更多的参数，导致更多的内存占用。


### create_tfrecords.py
```python3
import os
from PIL import Image
import tensorflow as tf


def create_record(path):
    cwd = os.getcwd()
    classes = os.listdir(cwd + path)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + path + name + "/"
        print(class_path)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((32, 32))
                image = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))
                writer.write(example.SerializeToString())
    writer.close()


def read_example():
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        label = example.features.feature['label'].int64_list.value
        print(label)


create_record("/train/")
# read_example()

```

### vgg19.py
```python3
import tensorflow as tf
import numpy as np


# print layer information
def print_layer(t):
    print(f"{t.op.name} {t.get_shape().as_list()} \n")


# conv layer op
def Conv2D(x, out, kernel_size, stride, name):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param kernel_size: kernel size.
    :param stride: step length.
    :param name: layer name.
    :return: activation.
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kernel_size, kernel_size, input_x, out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(x, kernel, (1, stride, stride, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        print_layer(activation)
        return activation


# define fully connected
def FullyConnected(x, out, name):
    """

    :param x: input tensor.
    :param out: output tensor.
    :param name: layer name.
    :return: activation
    """
    input_x = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[input_x, out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1, shape=[out], dtype=tf.float32, name='b'))
        activation = tf.nn.relu_layer(x, kernel, biases, name=scope)
        print_layer(activation)
        return activation


# define max pool layer
def MaxPool2D(input_op, kernel_size, stride, name):
    """

    :param input_op: input tensor.
    :param name: layer name
    :param kernel_size: kernel size.
    :param stride: step length.
    :return: tf.nn.max_pool.
    """
    return tf.nn.max_pool(input_op,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


# VGG neural network
def vgg19(images, keep_prob, classes):
    """

    :param images: input img tensor.
    :param keep_prob: dropout.
    :param classes: classifier classes.
    :return: pred classes.
    """
    conv1_1 = Conv2D(images, 64, kernel_size=3, stride=1, name='conv1_1')
    conv1_2 = Conv2D(conv1_1, 64, kernel_size=3, stride=1, name='conv1_2')
    pool1 = MaxPool2D(conv1_2, kernel_size=2, stride=2, name='max_pool1')

    conv2_1 = Conv2D(pool1, 128, kernel_size=3, stride=1, name='conv2_1')
    conv2_2 = Conv2D(conv2_1, 128, kernel_size=3, stride=1, name='conv2_2')
    pool2 = MaxPool2D(conv2_2, kernel_size=2, stride=2, name='max_pool2')

    conv3_1 = Conv2D(pool2, 256, kernel_size=3, stride=1, name='conv3_1')
    conv3_2 = Conv2D(conv3_1, 256, kernel_size=3, stride=1, name='conv3_2')
    conv3_3 = Conv2D(conv3_2, 256, kernel_size=3, stride=1, name='conv3_3')
    conv3_4 = Conv2D(conv3_3, 256, kernel_size=3, stride=1, name='conv3_4')
    # pool3 = MaxPool2D(conv3_4, kernel_size=2, stride=2, name='max_pool3')

    conv4_1 = Conv2D(conv3_4, 512, kernel_size=3, stride=1, name='conv3_1')
    conv4_2 = Conv2D(conv4_1, 512, kernel_size=3, stride=1, name='conv3_2')
    conv4_3 = Conv2D(conv4_2, 512, kernel_size=3, stride=1, name='conv3_3')
    conv4_4 = Conv2D(conv4_3, 512, kernel_size=3, stride=1, name='conv3_4')
    # pool4 = MaxPool2D(conv4_4, kernel_size=2, stride=2, name='max_pool3')

    conv5_1 = Conv2D(conv4_4, 512, kernel_size=3, stride=1, name='conv3_1')
    conv5_2 = Conv2D(conv5_1, 512, kernel_size=3, stride=1, name='conv3_2')
    conv5_3 = Conv2D(conv5_2, 512, kernel_size=3, stride=1, name='conv3_3')
    conv5_4 = Conv2D(conv5_3, 512, kernel_size=3, stride=1, name='conv3_4')
    pool5 = MaxPool2D(conv5_4, kernel_size=2, stride=2, name='max_pool3')

    flatten = tf.reshape(pool5, [-1, 4 * 4 * 512])
    fc6 = FullyConnected(flatten, 4096, name='fc6')
    dropout1 = tf.nn.dropout(fc6, rate=1 - keep_prob)

    fc7 = FullyConnected(dropout1, 4096, name='fc7')
    dropout2 = tf.nn.dropout(fc7, rate=1 - keep_prob)

    fc8 = FullyConnected(dropout2, classes, name='fc8')

    return fc8

```

### train.py
```python3
from datetime import datetime
from vgg19 import *

batch_size = 64
lr = 1e-4
classes = 10
max_steps = 50000


def read_and_decode(filename):
    """

    :param filename: tf records file name.
    :return: image and labels.
    """
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'data': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['data'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    # trans float32 and norm
    img = tf.cast(img, tf.float32)  # * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def train():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = vgg19(X, keep_prob, classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    images, labels = read_and_decode('train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=200,
                                                    min_after_dequeue=100)
    label_batch = tf.one_hot(label_batch, classes, 1, 0)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps):
            batch_x, batch_y = sess.run([img_batch, label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={X: batch_x, y: batch_y, keep_prob: 0.8})
            if i % 10 == 0:
                train_arr = accuracy.eval(feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
                print(f"{datetime.now()}: Step [%d/{max_steps}]  Loss : {i:.8f}, training accuracy :  {train_arr:.4g}")
            if (i + 1) == max_steps:
                saver.save(sess, './model/model.ckpt', global_step=i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()

```

文章引用于[marsjhao](https://blog.csdn.net/marsjhao/article/details/72955935) [zhangwei15hh](https://blog.csdn.net/zhangwei15hh/article/details/78417789)
编辑 [Lornatang](https://github.com/lornatang)
校准 [Lornatang](https://github.com/lornatang)
