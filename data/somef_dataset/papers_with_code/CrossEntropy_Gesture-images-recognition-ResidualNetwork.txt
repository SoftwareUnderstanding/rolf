# Gesture-images-recognition-ResidualNetwork
## 项目来源于Andrew Ng在Cousera上的公开课deeplearning.ai.
### 项目的要求是：对Andrew Ng提供的手势数据集进行识别, 图像内容为数字0-5的手势，图像为分辨率是64*64的RGB图像.
### 我的任务是搭建并训练了一个50层残差网络.ResidualNet是何凯明提出的一个强大的网络框架(https://arxiv.org/abs/1512.03385)
### 整个仓库分为三个大部分：
####  1.datasets----里面存放着训练集和测试集，均为h5文件. 其中训练集的样本个数为1080，0-6的手势图像各占360. 测试集的样本个数为120，0-6的手势图像各占20.  
####  2.Residula model----采用Keras搭建的网络. 
####  3.TensorFlow model----采用TensorFlow搭建的网络.
