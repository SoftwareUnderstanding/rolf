# metallic

### 项目结构说明

./

preprocess.ipynb --- 图片预处理+迁移模型的初步训练

./dataset --- 数据集

​				/train --- 训练数据

​				/test --- 测试数据

./utility

​				/\__init\__.py

​				/output.py --- 文件输出模块



### metrics.py介绍

关于metrics的计算方式

```python
computeMetrics(Ypred:list,Ytest:list) -> {'acc':acc...}
```



### output.py使用方式

- 初始化DataProcessor类时传入三个参数
  - 图片和标注的文件路径
  - 背景图片存放路径
  - 前景图片存放路径
- 调用process( ) 方法进行图片分割需依次传入四个参数
  - sliding window的长
  - sliding window的宽
  - sliding window x方向移动的步长
  - sliding window y方向移动的步长

#### eg

```python
crop = DataProcessor("./crop/", "dataset2/0/", "dataset2/1/")
crop.process(128, 128, 100, 100)
```



存放图片文件目录需如下所示

![image-20200423180449499](/Users/luyumin/Library/Application Support/typora-user-images/image-20200423180449499.png)



### 变更日志

##### [2020-4-26 12:15 zs commit to master]

###### 模型持久化

```python
#### 模型保存
def checkpoint(net:PyTorch-Model,model='se_resnet20'# model-name # ): None
#### 模型恢复
def modelrestore(model='se_resnet20' # model-name #): PyTorch-Model
```

###### Global变量

```python
#### 模型保存文件路径
MODEL_SAVE_PATH = './model'
#### 数据路径
DATA_PATH = r'E:\buffer\dataset\train'
```

###### Image Transformer

```python
#### image transformation
data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])
```

可以修改，但是要说明原因

###### 目前的模型

```python
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet20',
    num_classes=NUM_CLASS,
)
net = hub_model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=P_lr)
```

##### [2020-5-2 15:39 zs commit to master]

###### Global变量

```python
### 部分训练
_PARTIAL_TRAIN = True
_PARTIAL_TRAIN_RATIO = 0.003
```

部分训练数据比例

```python
### 冻结网络
_NET_FREEZE = True
_NET_NO_GRAD = []
```

冻结层 需要之后手动加层名

网络结构用 `utility.output.output_netork` 输出

这个以后慢慢来

###### 数据预处理

```python
#### image transformation for original images
data_transform_origin = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])

#### image transformation for augmented images
data_transform_aug = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])
```

`data_transform_origin`对应对原图像的变换

`data_transform_aug` 对应增强图像的变换

因为我的 `torchvision` 版本太低，到时候还需要加一些变换方法

EDataset是继承的数据集类，可以对应为 `ImageFolder` 对象

`getModel(NUM_CLASS,name='se_resnet50')` 是抽象出来的模型函数，方便以后修改

###### 模型优化

`lr_scheduler.StepLR(optimizer, SCHEDULE_EPOCH, SCHEDULE_REGRESS)`

用来控制学习率，参数之后可以再调

###### 模型分析

`utility.plot.plotResultCurve(_metrics:list,att_names:list,title='')`

作图函数，主要用来画评分

`utility.output.visualize_network`

原本想用tensorboard来可视化，但是我这里版本兼容不了，不知道你们行不行

`saveResult(_metrics:list,savePath:str)`

用来保存评分为 json 文件

对 `checkpoint(model, optimizer, epoch, useTimeDir=False)` 作了修改

能够按照时间来保存

###### 文件结构

test放的都是测试文件，垃圾内容

utility就是所有有用的模块

reference放大家自己的参考资料

model放模型结果

process放注释内容

log_res放日志文件

###### PS

大家有任何代码上的疑问可随时提问

##### [2020-5-3 11:12 zs commit to master]

###### 命令行参数

增加了argparser后可以使用命令行来执行，具体参数见`add_argument`

###### 参数保存

增加了参数保存方法（未测试）

###### 云文件管理

可以用 `process/upload.ipynb` 直接操作云端文件，具体方法在  `process/readme.md`

这些文件都能共享协作，比较方便

###### 模型参数

模型初步参数已经调整好了，在 `preprocess.ipynb` 中，可以直接运行

###### 文件结构

增加了analysis文件夹，用于后期模型的分析

增加了`train.py`用于执行服务器脚本







## Notes

### 损失函数

参考资料

- [pytorch loss function 总结](https://www.jianshu.com/p/579a0f4cbf24)

可以尝试的 loss

- KLDivLoss
- CrossEntropyLoss

### 初始化

参考资料

Xavier初始法论文：[http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf](https://link.zhihu.com/?target=http%3A//jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

He初始化论文：[https://arxiv.org/abs/1502.01852](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1502.01852)

### 优化

>adam,adadelta等,在小数据上,我这里实验的效果不如sgd, sgd收敛速度会慢一些，但是最终收敛后的结果，一般都比较好。如果使用sgd的话,可以选择从1.0或者0.1的学习率开始,隔一段时间,在验证集上检查一下,如果cost没有下降,就对学习率减半. 我看过很多论文都这么搞,我自己实验的结果也很好. 当然,也可以先用ada系列先跑,最后快收敛的时候,更换成sgd继续训练.同样也会有提升.据说adadelta一般在分类问题上效果比较好，adam在生成问题上效果比较好。



### Ensemble

> Ensemble是论文刷结果的终极核武器,深度学习中一般有以下几种方式
>
> - 同样的参数,不同的初始化方式
> - 不同的参数,通过cross-validation,选取最好的几组
> - 同样的参数,模型训练的不同阶段，即不同迭代次数的模型。
> - 不同的模型,进行线性融合. 例如RNN和传统模型.

### 参考资料

[你有哪些deep learning（rnn、cnn）调参的经验？](https://www.zhihu.com/question/41631631/answer/862075836)

