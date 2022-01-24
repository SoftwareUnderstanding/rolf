# ClassSR_paddle
## 一、简介
本项目采用百度飞桨框架paddlepaddle复现：ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic, by Jiaqing Zhang and Kai jiang (张佳青&蒋恺)
RCAN训练不太稳定，容易崩溃

paper：[ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)

code：[ClassSR](https://github.com/Xiangtaokong/ClassSR)

本代码包含了原论文的默认配置下的训练和测试代码。

## 二、复现结果

RCAN-branch1
| -      | Model | iteration | test5    |
| ------ | ----- | --------- | -------- |
| 原论文 | RCAN  | -         | 30.275dB |
| 复现   | RCAN  | 52.5w     | 30.281dB |

RCAN-branch2
| -      | Model | iteration | test5    |
| ------ | ----- | --------- | -------- |
| 原论文 | RCAN  | -         | 30.593dB |
| 复现   | RCAN  | 99w       | 30.492dB |

RCAN-branch3
| -      | Model | iteration | test5    |
| ------ | ----- | --------- | -------- |
| 原论文 | RCAN  | -         | 30.430dB |
| 复现   | RCAN  | 98w       | 30.178dB |

ClassSR-RCAN
| -                      | Model        | Test2K  | FLOPs         |
| ---------------------- | ------------ | ------- | ------------- |
| 原论文                 | ClassSR-RCAN | 26.39dB | 21.22G(65%)   |
| 复现 | ClassSR-RCAN | 26.38dB | 21.36(65.5%) |

![Results](https://github.com/icey-zhang/ClassSR_paddle/blob/main/results/ClassSR_result.png)

## 三、环境依赖

```
python -m pip install -r requirements.txt
```

此代码在python 3.7中进行了测试

## 四、实现

### 1. 测试
#### 1）下载数据集

下载处理好的数据集

   [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748)

    test2K.zip(测试集)

#### 2）下载Class_MODEL(Class_RCAN)的权重

下载权重latest_G.pdparams。[权重](https://pan.baidu.com/s/1wiIQb-dC-mBYFMJZVZa3pw) 提取码：i58k。

#### 3）修改路径

需要在[test_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/test/test_ClassSR_RCAN.yml)修改数据集路径、修改权重路径

#### 4）开始测试

```
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```

### 2. 训练 SR_MODEL(RCAN)
#### 1）下载数据集
下载处理好的数据集

[下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748) 

#### 2）修改路径

需要在[train_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/train)修改train_RCAN.yml（branch1）、train_RCAN2.yml（branch2）、train_RCAN3.yml（branch3）数据集路径   


####  3）开始训练

```
python train_ClassSR.py -opt options/train/train_RCAN.yml
```
```
python train_ClassSR.py -opt options/train/train_RCAN2.yml
```
```
python train_ClassSR.py -opt options/train/train_RCAN3.yml
```

### 3. 训练Class_MODEL(Class_RCAN)

#### 1）下载数据集

直接下载处理好的数据集

[下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748)
    
DIV2K_scale_sub.zip(训练集)
    
val_10.zip(验证集)

#### 2）修改路径

需要在[train_ClassSR_RCAN.yml](https://github.com/icey-zhang/ClassSR_paddle/blob/main/options/train/train_ClassSR_RCAN.yml)修改数据集路径，修改三个分支权重的路径

#### 3）开始训练

```
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```

## 五、代码结构


```
./ClassSR_paddle
├─data             
├─data_scripts                                          
├─models               #模型
├─options              #配置文件
├─results              #日志文件
├─utils                #一下API                                               
|  README.md                               
│  train.py            #分支训练
│  test.py             #分支测试
│  train_ClassSR.py    #ClassSR训练
│  test_ClassSR.py     #ClassSR测试

```

## 六、模型信息

| 信息     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 作者     | 张佳青                                                       |
| 时间     | 2021.08                                                      |
| 框架版本 | Paddle 2.1.2                                                 |
| 应用场景 | 图像超分                                                     |
| 模型权重 | [权重](https://pan.baidu.com/s/1wiIQb-dC-mBYFMJZVZa3pw) 提取码：i58k |
| 数据集   | [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/104667) [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/105748) [下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/55117) |
