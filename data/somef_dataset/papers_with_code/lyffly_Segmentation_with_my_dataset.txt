# Segmentation_with_my_dataset
will add more soon!

作者：刘云飞

建议与合作联系邮箱：liuyunfei.1314@163.com



### 0x00 语言和工具

语言：Python 3.7

框架：PyTorch 1.2

标注工具：Labelme

网络结构：U-Net

### 0x01 标注数据

数据标注采用labelme，如下为其标注界面，用点组成多边形。

![](imgs/labelme.png)



### 0x02 网络结构及训练

will add soon

论文名称：U-Net: Convolutional Networks for Biomedical Image Segmentation

U-net 论文地址：https://arxiv.org/abs/1505.04597

网络架构图（个人评价：简单实用不花梢）

![](imgs/unet.jpeg)

部分代码取之：https://github.com/JavisPeng/u_net_liver

### 0x03 结果

可以看到训练后，可以有效的区分前景和背景。

![](imgs/demo1.png)

在有遮挡的情况下，可以看到效果依旧很好。

![](imgs/demo2.png)

完整的Demo视频可以在B站观看，链接：https://www.bilibili.com/video/av77291164/

### 0x04 下一步

汽车车道的多种类物体分割