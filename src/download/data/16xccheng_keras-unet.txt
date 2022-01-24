# keras-unet-直肠癌分割

#### **运行环境**：**google colab**（免费拿谷歌的GPU来训练，前提是能**科学上网**）



#### 数据集来源：第七届“泰迪杯”数据挖掘挑战赛B题（题主也是参赛人员之一）

<http://www.tipdm.org/bdrace/tzjingsai/20181226/1544.html>



#### 文件结构：

##### ——keras-unet  # 保存处理后的图片

##### ——keras-unet2 #保存源图片文件，好多次处理图片

##### 代码放置于google colab中代码行中

![1555692046893](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\1555692046893.png)



#### 代码说明：代码分为3个文件（可以直接拷贝到google colab的代码行中运行修改）

##### ——before_data.py（先运行，处理训练集和测试集）

##### ——data_preparation.py（第二运行，将训练集和测试集的各种图片存储为矩阵并保存）

##### ——model_unet.py（最后，通过u-net模型进行训练分割图像）

##### 注：代码中路径采用绝对路径，运行前需要检查是否有正确的文件结构

#### 测试结果：

##### 代码默认使用train2作为训练集，共计1400多张图片，并选用test1中的1102作为测试集，其分割效果可以达到80%左右

##### 另外，model_unet还提供了wide u-net变体和u-net++变体模型，可以使用其进行训练测试。



##### 注：关于u-net的具体信息见论文<http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images>

#####        关于wide u-net和u-net++的信息见论文<https://arxiv.org/abs/1807.10165>

##### 在训练生成 .h5 文件后可以直接使用该文件进行测试

![1555692584581](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\1555692584581.png)



#### 至于后续预测分类部分题主还没弄完……（SVM，特征向量之类的……）
谢谢我的小考拉一直在身边支持我
