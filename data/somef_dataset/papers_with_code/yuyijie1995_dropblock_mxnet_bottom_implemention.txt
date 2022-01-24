# dropblock_mxnet_bottom_implemention
用C++实现一个mxnet版本dropblock Op 最后可以用mx.sym.Dropblock（）调用
dropblock_c++自定义OP实现 
GPU版本要调用的话，OP名改成了mx.sym.GPUDropblock()
===================================================================================================================
## 参考博客
[MXNet中新增Operator]（http://shuokay.com/2017/10/04/mxnet-add-op-in-backend/）
## 论文链接
[DropBlock: A regularization method for
convolutional networks]（https://arxiv.org/pdf/1810.12890.pdf）
## 参考python端dropblock实现代码
[pytorch实现]（https://github.com/miguelvr/dropblock）

[mxnet实现](https://github.com/chenzx921020/DropBlock-mxnet)
-----------------------------------------------------------------------------------------------------------------
### 已实现功能
* 3,5,7的blocksize的dropblock
* gpu，cpu均能进行训练
* mnist训练集上测试通过
* 新增了维度检查功能 用户输入二维特征图是程序不会奔溃
* 加入了p_keep schedule 功能
* 在cifar100上完成了测试 大约有1~1.5个百分点的提升

### dropblock_gpu和dropout_cpu mnist训练效果对比
![](https://github.com/yuyijie1995/dropblock_mxnet_bottom_implemention/blob/master/gpu_blocksize3.png)
![](https://github.com/yuyijie1995/dropblock_mxnet_bottom_implemention/blob/master/cpu_dropout.jpg)

### TODO
* 尝试增加高宽不相等的特征图的dropblock功能
* 学习谷歌代码规范对代码进行优化

