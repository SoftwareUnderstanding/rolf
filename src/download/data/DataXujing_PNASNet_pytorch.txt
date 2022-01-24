## Pytorch实现PNASNet([Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559))

## 基于ImageNet预训练模型的微调的迁移学习的实现

[arxiv](https://arxiv.org/abs/1712.00559)

**Xu Jing**

实现了：

+ 数据增强
    - 随机水平翻转
    - 随机竖直翻转
    - 随机亮度值(brightness)
    - 随机色调(hue)
    - 随机饱和度(saturation)
    - 随机对比度(contrast)

+ Pytorch加载训练集的pipeline
+ 基于ImageNet预训练的模型微调的PNASNet及训练
+ 单张图像和视频的推断

训练的参数设置：

+ batch size:16
+ epochs:300
+ Loss: CrossEntropyLoss
+ optim: Adam
+ lr: feature_param:0.0001, linear_param: 0.001
+ 硬件：ubuntu 16.04 64G, Tesla V100(32G)


模型训练:

```
python3 model.py
```

推断模型：

```
python3 inference.py
python3 inference_video.py
```

测试结果：

![](paper/test_res.png)
![](paper/test.png)

