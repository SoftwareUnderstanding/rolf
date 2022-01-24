# myDCGAN
参考文章：https://blog.csdn.net/stalbo/article/details/79359095
DCGAN论文地址：https://arxiv.org/pdf/1511.06434.pdf
我的代码是根据此项目修改学习的：https://github.com/carpedm20/DCGAN-tensorflow
# DCGAN
在读DCGAN([Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf))之前，我首先读的文章是GAN([Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf))
在以往的尝试中，将CNN应用于GAN，都没有获得成功。但是经过一系列探索，我们找到一类结构，可以在分辨率更高、更深的生成模型上稳定地训练。而本篇文章的方法是基于对CNN的以下改进：

 - **全卷积网络**（all convolutional net）：用步幅卷积（strided convolutions）替代确定性空间池化函数（deterministic spatial pooling functions）（比如最大池化）。允许网络学习自身 upsampling／downsampling方式（生成器G/判别器D）。在网络中，**所有的pooling层使用步幅卷积**（strided convolutions）(判别网络)和微步幅度卷积（fractional-strided convolutions）(生成网络)进行替换
 - 在卷积特征之上**消除全连接层**：例如：全局平均池化（global average pooling），曾被应用于图像分类任务（Mordvintsev et al.）中。global average pooling虽然可以提高模型的稳定性，但是降低了收敛速度。图1所示为模型的框架。
 - **批量归一化**（Batch Normalization）：将每个单元的输入都标准化为0均值与单位方差。这样有助于解决poor initialization问题并帮助梯度流向更深的网络。防止G把所有rand input都折叠到一个点。但是，将所有层都进行Batch Normalization，会导致样本震荡和模型不稳定，因此，生成器（G）的输出层和辨别器（D）的输入层不采用Batch Normalization
 - 激活函数：**在生成器（G）中，**输出层使用Tanh函数**，**其余层采用 ReLu 函数**** ; **判别器（D）中都采用leaky rectified activation**
 ![论文中的生成模型对于LSUN scene数据集的结构，在下面实验的对于celebA人脸数据集中对应的是512到64，而不是1024到128](https://img-blog.csdnimg.cn/20181116175605948.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpbmxpdXFpbg==,size_16,color_FFFFFF,t_70)上图是论文中的生成模型对于LSUN scene数据集的结构，在下面实验的对于celebA人脸数据集中对应的是512到64，而不是1024到128
 # DCGAN代码
 ## 数据集
 我用到的是CalebA人脸数据集（[官网链接](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)）是香港中文大学的开放数据，包含10,177个名人身份的202,599张人脸图片，并且都做好了特征标记，这对人脸相关的训练是非常好用的数据集。可以通过官网的百度云进行下载。其中img文件夹有三个文件，“img_align_celeba.zip”是jpg格式的，比较小，1G多，我采用的是这个文件，直接解压即可。其他文件夹的含义和标注可以网上搜索查阅。
 ## 代码结构
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181116181605608.png)

其中checkpoint是生成的模型保存的地方；logs是tensorboard --logdir logs来通过浏览器可视化一些训练过程；而samples是每训练100次patch后，验证generator输出的图片的保存文件夹；main是函数主入口，通过flags保存一系列参数；model是这个DCGAN的生成对抗性模型，而ops封装了一些model中调用的tensorflow的函数，方便调用，比如线性，反卷积（deconvolution）批量归一化（batch_norm）等；utils是一些图片处理保存之类的功能性函数，运行时将datadir改成你的celeA图片文件夹的地址，sample_dir改成celeA图片文件夹就可以直接运行，如果input_size和output_size要改，记住将utils中的也改掉
## 代码运行结果
记住生成的图片类似这种就算成功了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181116183857584.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FpbmxpdXFpbg==,size_16,color_FFFFFF,t_70)

这是我仅仅上传100个patch的结果，我之前训练一直没有人脸形状出现，幻想着也许多训练几轮就有效果，都是不可能的，如果你一开始都没有人脸的样子，那就要考虑是不是代码写错了 PS我刚开始是在tf.train.AdamOptimizer(..).minimize(self.d_loss,var_list=self.d_vars)中没有指定var_list，希望读者也注意，因为G和D网络在backpropagation时候如果不指定需要更新的参数，那会同时将两个网络的参数都进行更新，导致错误。
