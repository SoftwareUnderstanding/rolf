# 代码中的说明
---
## 1. 依赖项

- Python 3.6
- Numpy 1.14
- Pytorch 0.4
- Opencv 2.4
- VisualStudio 2013, CUDA8.0, cudnn6.0.
## 2. 文件结构

    | -- README.md
    | -- code
        | -- gen_data_csv.py
        | -- dataset.py
        | -- models.py
        | -- validation.py
        | -- train_net.py
        | -- test_net.py
    | -- data
        | -- guangdong_round1_test_a_20180916
        | -- guangdong_round1_test_b_20181009
        | -- guangdong_round1_train2_20180916
    | -- data_preprocess
        | -- RotImage.cpp
        | -- RotImage.h
    | -- submit
## 3. 引用

### 预训练模型

- [imagenet]
- [http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth)
### 训练模型

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
- [https://arxiv.org/abs/1602.07261](https://arxiv.org/abs/1602.07261)
### 模型说明

        我们以Inception-V4网络模型为主干，训练我们预处理之后的图像
## 4 图像预处理说明

### 预处理类作用

    我们把图像的预处理过程打包成C++的类，主要用于将输入图像中存在角度的铝型材图像进行旋转。
### 预处理步骤

    1）将原始图像转化到HSV空间中；
    2）对S通道进行二值化，并提取相应轮廓，计算轮廓外接矩形，获取其角度；
    3）根据角度，对原始图像进行旋转，将铝型材图像进行角度矫正；
    4）对旋转后的空白区域进行填充，得到最终结果。
<p align="center">
  <img width="800" height="728" src="./img/data_preprocess.jpg">
</p>


