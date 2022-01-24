# 1.	模型简介
  UNet++架构本质上是一个深度监督的编码器-解码器网络，其中编码器和解码器子网通过一系列嵌套的、密集的跳跃路径连接,重新设计的跳过路径旨在缩小编码器和解码器子网特征映射之间的语义差距.
##1.1网络模型结构简介
整体结构就是先编码（下采样）， 再解码（上采样），回归到跟原始图像一样大小的像素点的分类。

    a)	下采样
      i.	四个unit，两个3x3的conv加Relu为一个单位，然后进行2x2的maxpool，其中卷积不包括padding，每次尺寸会减少2，maxpool会将图片大小减半，通道数翻倍
      ii.	意义：降低图像平移旋转等操作带来的扰动，降低图像过拟合的风险，多层下采样得到的大量特征通道，能够保留像素周围的上下文信息，能感知一块一块的像素信息，而不是一个pixel
    b)	上采样
      i.	四个上采样的unit，2x2的反卷积，通道数减半，尺寸加倍
      ii.	Copy and crop 将同一层的下采样的结果和反卷积得到的结果通道叠加，注意反卷积得到的图片尺寸小于原来的图，故需要对原图有个裁剪操作。可以将浅层的定位信息和高层的像素分类的判定信息结合
    c)	1x1的卷积
       i.	将64通道转换为2通道，背景+目标的组织图像

## 1.2	数据集
本次实验数据集为高分遥感影像：

    来源：数据为1米分辨率的高分遥感影像，由广东南方数码科技股份有限公司采集、标注、构建；
    规模：5万张遥感影像耕地样本数据。训练集为2万张遥感影像和标注文件（公开），验证集为3万张遥感影像（非公开）。
    数据说明：原始影像格式为png，包含R、G、B三个波段，训练集影像尺寸为256 * 256像素，测试集影像尺寸为256 * 256像素；标签格式为单通道的png，0表示为非耕地，1表示为耕地。

在实验中，首先将数据集转换为Multi-Class格式，存放在unet/dataset目录下，通过prepare.py将OBS上的数据下载至本地并划分为Multi-Class格式。
Multi-Class数据集格式，通过固定的目录结构获取图片和对应标签数据。在同一个目录中保存原图片及对应标签，其中图片名为“image.png”，标签名为“mask.png”。目录结构如下：
```.
└─dataset
  └─0001
    ├─image.png
    └─mask.png
  └─0002
    ├─image.png
    └─mask.png
    ...
  └─xxxx
    ├─image.png
    └─mask.png
```

## 1.3 码交址代码提交地址
https://github.com/Tu-kun/cultivated_landdivision

## 1.4模型大小
本次提交的代码，支持在Asecnd环境下训练，模型大小为23M.
 
# 2.	代码目录结构说明
```
├── eval.py                                       // 评估脚本
├── export.py                                     // 模型导出
├── prepare_data.py                           // 适配Multiclass数据集
├── preprocess.py                                   // 310推理预处理脚本
├── run.ipynb                                    // 模型训练，验证过程记录
├── requirements.txt                                // 模型依赖
├── dataset                                        // 训练数据所在目录
├── src
│   ├── data_loader.py                              // 数据处理
│   ├── eval_callback.py                            // 训练时推理回调
│   ├── __init__.py
│   ├── loss.py                                     // loss函数
│   ├── model_utils
│   │   ├── config.py                               // 配置文件
│   │   ├── device_adapter.py                       // 设备适配
│   │   ├── __init__.py
│   │   ├── local_adapter.py
│   │   └── moxing_adapter.py                       // modelarts适配
│   ├── unet_nested
│   │   ├── __init__.py
│   │   ├── unet_model.py                           // Unet++网络结构
│   │   └── unet_parts.py                           // Unet++子网
│   └── utils.py
├── train.py                                        // 训练脚本
├── unet.yaml                        		// unet配置文件
└── version.ini                                     // 版本信息
```


# 3.	自验结果
##3.1自验环境
华为云modelarts平台，Ascend910   
镜像环境为：tensorflow1.15-mindspore1.3.0-cann5.0.2-euler2.8-aarch64

## 3.2 训练超参数
模型训练的一些参数如下：

    lr: 0.001            # 学习率
    epochs: 400         # 迭代次数
    batch_size: 32       # 批处理大小
    cross_valid_ind: 5    # 交叉验证次数
    num_classes: 2      # 分类个数
    num_channels: 3    # 输入图像通道数
## 3.3 训练
模型训练
### 3.3.1 启动训练
```python train.py --data_path=dataset/ --config_path=unet.yaml --output ./output```
### 3.3.2 训练精度结果
在华为的ModerArts Ascend910平台上，训练中最大FWIou的值为0.39914
![](doc/result.png)  
验证精度（此处数据集较小）  
![](doc/val.jpg)  
 
## 3.4 模型评估
```
# 将验证数据集更改为Multi-Class数据集格式
python prepare_data.py train=False -datapath=训练集文件路径 -dstPath=更改格式后的验证文件存放目录
# 验证代码
python eval.py –data_path=更改格式后的验证文件存放目录  -checkpoint_file_path=output/checkpoint/ckpt_o/best.ckpt  
```
# 4. 参考资料
在比赛过程中我们参考了一些资料，很感谢论文作者以及华为的MindSpore框架以及ModelZoo中各种开源的模型库。

## 4.1 参考论文
[1] Ronneberger O ,  Fischer P ,  Brox T . U-Net: Convolutional Networks for Biomedical Image Segmentation[J]. Springer International Publishing, 2015.
（https://arxiv.org/abs/1505.04597）
[2] Zhou Z ,  Siddiquee M ,  Tajbakhsh N , et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation[C]// 4th Deep Learning in Medical Image Analysis (DLMIA) Workshop. 2018.（https://arxiv.org/pdf/1807.10165.pdf）

## 4.2 参考git
https://github.com/4uiiurz1/pytorch-nested-unet

