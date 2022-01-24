## 机器学习大作业

作业的目的：

1、在X光的检测图像中，定位到充电宝的位置【检测】，并判断有无带电芯【分类】。

2、评价标准 ： mAP

#### 近期进度

1、相关知识和文献：

​	1）faster R-CNN:

​		 https://zhuanlan.zhihu.com/p/31426458     

​		 https://zhuanlan.zhihu.com/p/49897496

​		 https://arxiv.org/abs/1506.01497 		

​	2）不平衡分类：

​		 focal loss 和 GHM：

​			https://zhuanlan.zhihu.com/p/80594704

​		不过总觉得知乎上讨论的样本不均衡和我们说的不是一回事。。

​		这些论文讨论的正负样本指的是：

**正样本：标签区域内的图像区域，即目标图像块**

**负样本：标签区域以外的图像区域，即图像背景区域**

3）在二阶段法的分类中，如果不平衡，似乎是通过调整权重、阈值等方法，这个还值得进一步的考查

4）参考源码：

 https://github.com/longcw/faster_rcnn_pytorch

https://github.com/chenyuntc/simple-faster-rcnn-pytorch

​

#### 近期任务

1）阅读faster R-CNN 的相关论文，了解二阶段的具体流程，了解RPN层的原理，了解Classification的具体实现[之后可能要改造的部分]

2）在windows上跑通上述源码的demo，为之后对该代码进行瘦身和改造做铺垫

3）对数据集进行基本的预处理，生成基本的样本读取文档



#### 进度记录

**12.1**

已完成：

1）对图片进行填充或裁剪，成为标准的2000 * 1040 大小 - ch

2）生成初步的文件读入list - zwj

3） 参照 simple-faster-rcnn 源码，改造dataset部分 - zyz

待考察：

1） focal loss的使用

2） roi pooling的原理以及通过cupy的编译



12.13

已完成：

采用预训练模型，但由于类别数量与预训练模型不同，所以删去预训练中的最后一层的参数

同时，对于样本不均衡，采用了适度的过采样的方法，将训练集中在core样本，复制了两份

模型run通，训练10轮，通过4:1的数据集划分，在验证集中，map最高可达0.77





#### 测试说明

1、将测试文件的id_list文件放在data目录下，将测试的image和annotation放到对应的文件夹中

2、链接：https://pan.baidu.com/s/1W9CPgou46-F_AdSc7iZAOA 提取码：nbeq
   将以上链接中的文件（fasterrcnn_12121726_ep3）放入checkpoints目录下

3、运行以下命令：其中testfile为id_list的文件名，请确保id_list中的文件名能和annotation中在文件名完全对应

```shell
python train.py train  --istrain=False  --testfile='test.txt'
```
