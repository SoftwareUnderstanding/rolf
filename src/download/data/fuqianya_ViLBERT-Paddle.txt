# ViLBERT-Paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)实现

## 一、简介

本项目使用[paddle](https://github.com/PaddlePaddle/Paddle)框架复现[ViLBERT](https://arxiv.org/abs/1908.02265)模型。该模型包含两个并行的流，分别用于编码visual和language，并且加入了 co-attention transformer layer 来增强两者之间的交互，得到 pretrained model。作者在多个 vision-language 任务上得到了多个点的提升。

**注: AI Studio项目地址: [https://aistudio.baidu.com/aistudio/projectdetail/2609611](https://aistudio.baidu.com/aistudio/projectdetail/2609611).**

**您可以使用[AI Studio](https://aistudio.baidu.com/)平台在线运行该项目!**

**论文:**

* [1] J. Lu, D. Batra, D. Parikh, S. Lee, "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks", NIPS, 2019.

**参考项目:**

* [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task) [官方实现]

## 二、复现精度

> 所有指标均为模型在[RefCOCO+](https://arxiv.org/abs/1608.00272)的验证集评估而得

| 指标 | 原论文 | 复现精度 | 
| :---: | :---: | :---: | 
| Acc | 72.34 | 72.71 |

## 三、数据集

本项目所使用的数据集为[RefCOCO+](https://arxiv.org/abs/1608.00272)。该数据集共包含来自19,992张图像的49,856个目标对象，共计141,565条指代表达。本项目使用作者提供的预提取的`bottom-up`特征，可以从[这里](https://www.dropbox.com/sh/4jqadcfkai68yoe/AADHI6dKviFcraeCMdjiaDENa?dl=0)下载得到（我们提供了脚本下载该数据集以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/ViLBERT-Paddle/blob/main/download_dataset.sh)）。


## 四、环境依赖

* 硬件：CPU、GPU

* 软件：
    * Python 3.8
    * PaddlePaddle == 2.1.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/fuqianya/ViLBERT-Paddle.git
cd ViLBERT-Paddle
```

### step2: 安装环境及依赖

```bash
pip install -r requirements.txt
```

### step3: 下载数据

```bash
# 下载数据集
bash ./download_dataset.sh

# 下载paddle格式的预训练模型
# 放于checkpoints/bert_base_6_layer_6_connect_freeze_0下
# 下载链接: https://drive.google.com/file/d/1QMJz5anz_git8NFThUgacOBgYti_of4g/view?usp=sharing

# 编译REFER
cd pyutils/refer && make
cd ..
```

### step4: 训练

```bash
python train.py --gradient_accumulation_steps 1
```

Change `gradient_accumulation_steps` to adapt your GPU memory.

### step5: 测试


```bash
python eval.py --from_pretrained ./checkpoints/refcoco+_bert_base_6layer_6conect-pretrained/paddle_model_19.pdparams
```

### 使用预训练模型进行预测

模型下载: [谷歌云盘](https://drive.google.com/file/d/19gbGuVm9hgVPm_XzAUrTpeDmObr5ZAv3/view?usp=sharing)

将下载的模型权重以及训练信息放到`checkpoints/refcoco+_bert_base_6layer_6conect-pretrained`目录下, 运行`step5`的指令进行测试。

## 六、代码结构与详细说明

```bash
├── checkpoints     　   # 存储训练的模型
├── config             　# 配置文件
├── data            　   # 预处理的数据
├── model
│   └── vilbert.py    　 # 模型
│   └── rec_dataset.py 　# 加载数据集
│   └── optimization.py　# 定义优化器
├── result            　 # 存放预测结果
├── pyutils 
│   └── refer          　# REFER
├── utils 
│   └── io.py          　# io工具
│   └── eval_utils.py  　# 测试工具
│   └── utils.py       　# 其他工具
├── download_dataset.sh　# 数据集下载脚本
├── train.py           　# 训练主函数
├── eval.py            　# 测试主函数
└── requirement.txt   　 # 依赖包
```

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| :---: | :---: |
| 发布者 | fuqianya |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.0 |
| 应用场景 | 多模态 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/file/d/19gbGuVm9hgVPm_XzAUrTpeDmObr5ZAv3/view?usp=sharing) \| [训练日志](https://drive.google.com/file/d/1hwXfZUy3V2YnsBKQkQADvACTyXYqvLFa/view?usp=sharing)  |
