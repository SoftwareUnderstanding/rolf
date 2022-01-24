# Mlp-Mixer论文复现

​		Mlp-mixer是谷歌5月份提出的基于纯mlp结构的cv框架，用来替代卷积和Transformer里的注意力操作。

​		本项目用b16规模的mixer，使用在imagenet和imagenet21k上预训练的权重，在cifar10数据集上验证准确率分别达到了96.8%（2epoch）和97.18%（2epoch）。

![curve](./imgs/curve.png)

![acc](./imgs/acc.png)

## Tree

```
# 目录结构
/paddle
├── align.py # 权重对齐
├── ckpt/ # 两个预训练模型
├── conf
│   └── base.yaml # 配置文件
├── main.py # 运行
├── models/ # mixer代码
├── run.sh # 运行
├── scrips.py # 加载数据、训练、评估
└── utils/ # 日志
```

## Train

```
python main.py --config ./conf/base.yaml --mode train
或
./run.sh 0
```

## Evaluate

```
python main.py  --config ./conf/base.yaml --mode eval
或
./run.sh 1
```

## Link

**注：**

1.换预训练权重需要修改yaml配置文件里的model name（1k或21k）；

2.权重地址：

- imagenet预训练权重
  - 百度网盘链接：https://pan.baidu.com/s/1sLPrOM4WXq2SG23yxWtTeA  提取码：zm5v
  - 放到ckpt下
- cifar10迁移训练权重
  - 百度网盘链接: https://pan.baidu.com/s/13drJv02mF_FGWD-1sbACeQ 提取码: yrsc 

3.可以移步aistudio直接运行：

[aistudio Mlp-Mixer Paddle 复现](https://aistudio.baidu.com/aistudio/projectdetail/2258020)

另：

[Mlp-Mixer论文地址](https://arxiv.org/pdf/2105.01601v4.pdf)

[csdn:Mlp-Mixer简介](https://blog.csdn.net/weixin_43312063/article/details/117250816?spm=1001.2014.3001.5501)

