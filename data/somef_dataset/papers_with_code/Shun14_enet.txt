## ENet paddle version
基于paddleSeg实现了[ENet](https://arxiv.org/abs/1606.02147)

0. pip install -r requirements.txt 安装所需安装包
1. 原始论文采用了300epochs训练，我使用了batch为8，训练80k steps的思路，实际并没有train到300epochs，120k的配置文件则train到了300epoch以上，训练尺寸与原论文一致，具体参数请参考配置文件，其中测试均在cityscapes val dataset进行，与原始论文保持一致，80k设置模型最高为58.3，120k的设置最高为60.3，均达到标准，配置文件中enet_baseline_cityscapes_1024x512_adam_0.002_80k_weight的weight为提取出来的cityscapes类别权重，不需要使用即可得到原论文结果
2. 将cityscapes路径复制到datasets路径下，同时运行tools/convet_cityscapes.py 得到可训练测试数据，运行run.sh即可训练，运行run_eval.sh即可测试80k模型得到结果，修改模型路径可以修改run_eval.sh文件，模型和日志在百度网盘中
3. 训练log和训练权重[百度网盘地址](https://pan.baidu.com/s/1Qx9sfL0t1z7FTqTLeRCJ-A)，其中包括80k权重位置和120k权重位置
