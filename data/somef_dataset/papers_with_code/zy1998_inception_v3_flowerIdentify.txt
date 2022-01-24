# inception_v3_flowerIdentify
使用inceptionV3模型训练识别5种花朵

## 项目介绍
 
在tensorflow框架下, 利用谷歌的inceptionV3模型训练自己采集的数据集，实现五种花朵（月季、绣球、万寿菊、三色堇、石榴花）的识别。
tonado搭建后端, h5+css+js完成前端页面。

## InceptionV3介绍
InceptionV3网络是基于InceptionV2上的再次改造，主要在两个方面。一方面引入了Factorization into small convolutions的思想，将一个较大的二维卷积拆成两个较小的一维卷积，比如将7x7拆成1x7和7x1，将3x3拆成1x3和3x1，节约了大量参数，加速运算并减轻了过拟合，同时增加了一层非线性扩展模型表达能力。论文中指出，这种非对称的卷积结构拆分，其结果比对称地拆分为几个相同的小卷积核效果更明显，可以处理更多，更丰富的空间特征，增加特征多样性。<br>
另一方面，inceptionV3优化了Inception Module的结构，在其中使用了分支，分支里面还使用了分支，可以说，Network In Network In Network。<br>
论文地址*https://arxiv.org/abs/1512.00567*

## 运行
![image](https://github.com/zy1998/inception_v3_flowerIdentify/blob/master/static/images/run.JPG)


## 预览图
模型验证的准确率: <br>
![image](https://github.com/zy1998/inception_v3_flowerIdentify/blob/master/static/images/%E5%87%86%E7%A1%AE%E7%8E%87.JPG) <br>
手机端拍照识别"月季花"的结果: <br>
![image](https://github.com/zy1998/inception_v3_flowerIdentify/blob/master/static/images/%E6%9C%88%E5%AD%A3%E8%8A%B1%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.png) <br>
手机端拍照识别"绣球花"的结果: <br>
![image](https://github.com/zy1998/inception_v3_flowerIdentify/blob/master/static/images/%E7%BB%A3%E7%90%83%E8%8A%B1%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.png) <br>
手机端拍照识别"石榴花"的结果: <br>
![image](https://github.com/zy1998/inception_v3_flowerIdentify/blob/master/static/images/%E7%9F%B3%E6%A6%B4%E8%8A%B1%E8%AF%86%E5%88%AB%E7%BB%93%E6%9E%9C.png)



## 作者与联系方式
*Yu Zeng* <br>
*2194877791@qq.com*
