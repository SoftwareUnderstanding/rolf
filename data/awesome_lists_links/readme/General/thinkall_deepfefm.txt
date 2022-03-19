# deepfefm
本项目为基于PaddlePaddle 2.1.0 复现 DeepFeFm 论文，基于 [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 二次开发。

论文地址：[Field-Embedded Factorization Machines for Click-through rate prediction](https://arxiv.org/pdf/2009.09931v2.pdf)

原论文代码：[DeepCTR-DeepFEFM](https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/deepfefm.py)

在线使用：[AIStudio在线体验项目](https://aistudio.baidu.com/aistudio/projectdetail/2253037)

## 模型简介
该模型是FM类模型的又一变种。模型架构如下：

<div align="center"><img src=https://i.loli.net/2021/08/13/jdbOnL5zH4IPu2V.png height=500></img></div>

模型的核心公式如下：

<div align="center"><img src=https://i.loli.net/2021/08/13/XVBLZ89nI3SrDt7.png height=40></img></div>

其中与FFM，FwFM模型的核心区别就是使用一个对称矩阵 Field pair matrix embeddings `$W_{F(i),F(j)}$` 对不同field的关系进行建模。

## 复现精度

| 数据集 | 复现精度 |
| --- | --- |
| [Criteo(Paddle版)](https://github.com/PaddlePaddle/PaddleRec/blob/master/datasets/criteo/run.sh) | 0.80276 |

- 核心参数设置
```
- lr: 0.0005
- batch_size: 5120
- optimizer: Adam
```

## 代码简介

```
├── LICENSE                                   协议文本
├── README.md                                 项目简介
├── config                                    配置文件夹
│   ├── config.yaml                           小样本配置文件
│   └── config_bigdata.yaml                   全量数据配置文件
├── data                                      样本数据文件夹
│   └── sample_data
│       └── train
│           └── sample_train.txt
├── papers                                    论文
│   └── 2009.09931v2.pdf
└── src
    ├── __init__.py
    ├── criteo_reader.py                      数据读取代码
    ├── dygraph_model.py                      模型构造代码
    ├── infer.py                              推理代码
    ├── myutils.py                            工具算子代码
    ├── net.py                                核心组网代码
    ├── preprocess_raw_data.py                数据探索代码
    ├── trainer.py                            训练代码
    └── utils                                 PaddleRec工具代码
        ├── envs.py
        ├── save_load.py
        └── utils_single.py
```

## 使用方法
### 启动训练
- 样例数据
```
python -u src/trainer.py -m config/config.yaml
```
- 全量数据
```
python -u src/trainer.py -m config/config_bigdata.yaml
```

- 部分训练日志：
```
2021-08-13 15:52:32,885 - INFO - **************common.configs**********
2021-08-13 15:52:32,885 - INFO - use_gpu: True, use_visual: False, train_batch_size: 5120, train_data_dir: /home/aistudio/data/slot_train_data_full, epochs: 4, print_interval: 100, model_save_path: output_model_all_deepfm, save_checkpoint_interval: 1
2021-08-13 15:52:32,885 - INFO - **************common.configs**********
W0813 15:52:32.886739  1867 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0813 15:52:32.891144  1867 device_context.cc:422] device: 0, cuDNN Version: 7.6.
2021-08-13 15:52:37,460 - INFO - read data
2021-08-13 15:52:37,461 - INFO - reader path:criteo_reader
2021-08-13 15:52:37,461 - INFO - reader path:criteo_reader
2021-08-13 15:52:42,372 - INFO - epoch: 0, batch_id: 0, auc:0.500132, loss:[0.6964508], avg_reader_cost: 0.01165 sec, avg_batch_cost: 0.04896 sec, avg_samples: 51.20000, ips: 1045.83222 ins/s, loss: 0.696451
2021-08-13 15:55:52,142 - INFO - epoch: 0, batch_id: 100, auc:0.682062, loss:[0.50682473], avg_reader_cost: 0.00037 sec, avg_batch_cost: 1.89720 sec, avg_samples: 5120.00000, ips: 2698.71022 ins/s, loss: 0.506825
2021-08-13 15:58:56,466 - INFO - epoch: 0, batch_id: 200, auc:0.716817, loss:[0.47779626], avg_reader_cost: 0.00032 sec, avg_batch_cost: 1.84275 sec, avg_samples: 5120.00000, ips: 2778.44940 ins/s, loss: 0.477796
2021-08-13 16:02:01,150 - INFO - epoch: 0, batch_id: 300, auc:0.733832, loss:[0.4546355], avg_reader_cost: 0.00031 sec, avg_batch_cost: 1.84641 sec, avg_samples: 5120.00000, ips: 2772.95353 ins/s, loss: 0.454636
.
.
.
2021-08-13 20:11:17,361 - INFO - epoch: 0, batch_id: 8200, auc:0.795808, loss:[0.45324177], avg_reader_cost: 0.00033 sec, avg_batch_cost: 1.88111 sec, avg_samples: 5120.00000, ips: 2721.79535 ins/s, loss: 0.453242
2021-08-13 20:14:28,855 - INFO - epoch: 0, batch_id: 8300, auc:0.795979, loss:[0.4493463], avg_reader_cost: 0.00031 sec, avg_batch_cost: 1.91445 sec, avg_samples: 5120.00000, ips: 2674.40082 ins/s, loss: 0.449346
2021-08-13 20:17:35,016 - INFO - epoch: 0, batch_id: 8400, auc:0.796083, loss:[0.43260103], avg_reader_cost: 0.00040 sec, avg_batch_cost: 1.86115 sec, avg_samples: 5120.00000, ips: 2750.99465 ins/s, loss: 0.432601
2021-08-13 20:20:42,364 - INFO - epoch: 0, batch_id: 8500, auc:0.796201, loss:[0.42714283], avg_reader_cost: 0.00034 sec, avg_batch_cost: 1.87298 sec, avg_samples: 5120.00000, ips: 2733.61228 ins/s, loss: 0.427143
2021-08-13 20:23:30,465 - INFO - epoch: 0 done, auc: 0.796279,loss:[0.44033897], epoch time: 16253.00 s
2021-08-13 20:23:33,390 - INFO - Already save model in output_model_all_deepfm/0
```

### 评估模型
- 样例数据
```
python -u src/infer.py -m config/config.yaml
```
- 全量数据
```
python -u src/infer.py -m config/config_bigdata.yaml
```

- 验证日志：
```
2021-08-13 20:56:21,228 - INFO - **************common.configs**********
2021-08-13 20:56:21,228 - INFO - use_gpu: True, use_xpu: False, use_visual: False, infer_batch_size: 5120, test_data_dir: /home/aistudio/data/slot_test_data_full, start_epoch: 0, end_epoch: 4, print_interval: 100, model_load_path: output_model_all_deepfm
2021-08-13 20:56:21,228 - INFO - **************common.configs**********
W0813 20:56:21.229516 54904 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0813 20:56:21.233371 54904 device_context.cc:422] device: 0, cuDNN Version: 7.6.
2021-08-13 20:56:25,746 - INFO - read data
2021-08-13 20:56:25,747 - INFO - reader path:criteo_reader
2021-08-13 20:56:25,750 - INFO - load model epoch 0
2021-08-13 20:56:25,750 - INFO - start load model from output_model_all_deepfm/0
2021-08-13 20:56:31,285 - INFO - epoch: 0, batch_id: 0, auc: 0.790873, avg_reader_cost: 0.01153 sec, avg_batch_cost: 0.04754 sec, avg_samples: 5120.00000, ips: 92505.88 ins/s
2021-08-13 20:58:59,654 - INFO - epoch: 0, batch_id: 100, auc: 0.802344, avg_reader_cost: 0.02974 sec, avg_batch_cost: 1.48363 sec, avg_samples: 5120.00000, ips: 3450.89 ins/s
2021-08-13 21:01:21,692 - INFO - epoch: 0, batch_id: 200, auc: 0.802837, avg_reader_cost: 0.00072 sec, avg_batch_cost: 1.42019 sec, avg_samples: 5120.00000, ips: 3605.05 ins/s
2021-08-13 21:03:43,941 - INFO - epoch: 0, batch_id: 300, auc: 0.802896, avg_reader_cost: 0.03356 sec, avg_batch_cost: 1.42232 sec, avg_samples: 5120.00000, ips: 3599.65 ins/s
2021-08-13 21:05:02,232 - INFO - epoch: 0 done, auc: 0.802757, epoch time: 516.48 s
```

- 预训练模型

链接: https://pan.baidu.com/s/1CftnEt0nl1V6w6ApDzqkKA 提取码: dyvf

## 参数简介
### 6.2 参数说明
通过`config/*.yaml`文件设置训练和评估相关参数，具体参数如下：
|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
|runner.train_data_dir|"data/sample_data/train"|训练数据所在文件夹|
|runer.train_reader_path|"criteo_reader"|训练数据集载入代码|
|runer.use_gpu|True|是否使用GPU|
|runer.train_batch_size|5120|训练时batch_size|
|runer.epochs|1|训练几个epoch|
|runner.print_interval|50|多少个batch打印一次信息|
|runner.model_init_path|"output_model_dmr/0"|继续训练时模型载入目录，默认未启用|
|runner.model_save_path|"output_model_dmr"|模型保存目录|
|runner.test_data_dir|"data/sample_data/test"|测试数据文件夹|
|runner.infer_reader_path| "alimama_reader"|测试数据集载入代码|
|runner.infer_batch_size|256|评估推理时batch_size|
|runner.infer_load_path|"output_model_dmr"|评估推理时模型载入目录|
|runner.infer_start_epoch|1000|评估推理的从哪个epoch开始，默认会把最优模型保存到目录1000，所以默认从1000开始，当然也可以从0开始|
|runner.infer_end_epoch|1001|评估推理到哪个epoch（不含）停止，默认值实际上只评估1000目录中的这1个模型|
|hyper_parameters.optimizer.class|Adam|优化器，Adam效果最好|
|hyper_parameters.optimizer.learning_rate|0.0005|学习率，应与batchsize同步调整|
|hyper_parameters.sparse_feature_dim|48|Embedding长度|
|hyper_parameters.fc_sizes|[1024, 1024, 1024]|隐藏层大小|

## 模型信息
关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 |bnujli |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.0 |
| 应用场景 | 点击率预测|
| AUC |  0.802757  |
