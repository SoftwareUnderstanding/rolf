# 使用PaddlePaddle复现论文：Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer

## T5

[Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3.pdf)

**Abstract**
Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language
processing (NLP). The effectiveness of transfer learning has given rise to a diversity of
approaches, methodology, and practice. In this paper, we explore the landscape of transfer
learning techniques for NLP by introducing a unified framework that converts all text-based
language problems into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other factors on
dozens of language understanding tasks. By combining the insights from our exploration
with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results
on many benchmarks covering summarization, question answering, text classification, and
more. To facilitate future work on transfer learning for NLP, we release our data set,
pre-trained models, and code.1

本项目是 T5 在 Paddle 2.x上的开源实现。

## 原论文效果
<p align="center">
    <img src="figure/paper_result.jpg" width="100%" />
</p>


## 环境安装

| 名称   | 值             |
|--------|------------------|
| python | 3\.8             |
| GPU    | RTX3090          |
| 框架    | PaddlePaddle2\.1 |
| Cuda   | 11\.2            |
| Cudnn  | 8\.1\.1\.33\-1   |

或者使用本次复现使用的云平台：https://gpushare.com/
<p align="center">
    <img src="figure/yunfuwuqi.jpg" width="100%" />
</p>

```bash
# 克隆本仓库
git clone https://github.com/junnyu/paddle_t5
# 进入paddlenlp目录
cd paddlenlp
# 本地安装
pip install -r requirements.txt
pip install -e .
# 返回初始目录
cd ..
```

## 快速开始


### （一）模型精度对齐
运行`python compare.py`，对比huggingface与paddle之间的精度，我们可以发现精度的平均误差在10^-7量级，最大误差在10^-5量级。
```python
python compare.py
# t5-small
# mean difference: tensor(1.3692e-07)
# max difference: tensor(1.9073e-05)
# t5-base
# mean difference: tensor(2.2535e-07)
# max difference: tensor(1.5259e-05)
# t5-large
# 内存不够（转换代码一样，结果差不多的）
```
#### 转化后成paddle的t5模型连接，small,base,large（当然也可以手动转换，`python convert.py`）
链接：https://pan.baidu.com/s/1f4o_A9FYjcEOxxoqcVbMSQ 
提取码：48se 



### （二）下游任务微调

#### 1、GLUE
以MNLI数据集为例（对于其他GLUE任务，请参考`TASK/GLUE/train.sh`，该脚本有任务的训练参数）

##### （1）模型微调：
```shell
# 确保处在GLUE文件夹
cd TASK/GLUE
# 安装依赖
pip install -r requirements.txt
# 运行训练
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name mnli \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir outputs/mnli/ \
    --device gpu \
    --num_workers 2
```
其中参数释义如下：
- `model_name_or_path` 模型名称或者路径。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE、 WNLI。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `train_batch_size` 表示训练阶段每次迭代的样本数目。
- `eval_batch_size` 表示评估阶段每次迭代的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `scheduler_type` scheduler类型，可选linear和cosine，默认linear。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU。
- `seed` 随机种子。
- 还有其他的参数就不介绍了。

**模型链接**(这个链接包含所有GLUE任务微调后的权重)

链接：https://pan.baidu.com/s/1Nwqazg1bzW8N-gIWe2BbfQ 
提取码：d5b6 


##### （2）模型预测及提交(TODO)：
**等有时间再预测，然后提交到GLUE排行榜。**


###### GLUE开发集结果：
| Model                          | cola  | sst-2  | mrpc        | sts-b             | qqp         | mnli       | qnli | rte   | mean |
|--------------------------------|-------|-------|-------------|------------------|-------------|-------------|------|-------|-------|
|                                | mcc   | acc   | acc      | pearson | acc      | acc      | acc  | acc   |         |
| T5-base-Paddle | 61.74 | 95.18 | 90.44 | 90.09   | 91.60 | 87.18 | 93.56 | 81.95 | 86.4675 |

###### 查看训练日志：
```bash
visualdl --logdir logs/GLUE
```
<p align="center">
    <img src="figure/GLUE-keshihua.jpg" width="100%" />
</p>


#### 2、CNNDM

使用`T5-base`预训练模型运行`CNNDM摘要数据集`的Fine-tuning。https://huggingface.co/datasets/cnn_dailymail


##### （1）准备数据：
由于paddlenlp没有该数据集，我从huggingface使用datasets加载该数据集，然后保存为json，再然后自定义paddlenlp的dataset数据集进行训练。
注意：我自定义了`paddlenlp/paddlenlp/datasets/cnn_dailymail.py`，我改了代码，因此这个没从网站下载数据，直接使用本地的json。

##### 手动转换（数据集很大，训练集1个多G，验证测试集分别50MB）：
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail","3.0.0")
dataset["train"].to_json("caches/cnndailymail/cnn_dailymail_train.json")
dataset["validation"].to_json("caches/cnndailymail/cnn_dailymail_dev.json")
dataset["test"].to_json("caches/cnndailymail/cnn_dailymail_test.json")
```

##### 或者直接下载转换好的数据集（json格式）：
链接：https://pan.baidu.com/s/1VSRsB8GTXH1cBqXcUZo6CQ 
提取码：q0cd 

##### aistudio链接：
这个是在aistudio训练的，代码有点乱，不过差不多了，结构一样。
https://aistudio.baidu.com/aistudio/projectdetail/2300668。

```shell
# 确保处在CNNDM文件夹
cd TASK/CNNDM
# 安装依赖
pip install -r requirements.txt
# 开始训练，注意修改预训练模型路径，其他参数我都固定好了。
python run_cnn_dailymail.py \
    --model_name_or_path data/t5-base
```
###### 查看训练日志：
```bash
visualdl --logdir logs/CNNDM
```
<p align="center">
    <img src="figure/CNNDM-keshihua.jpg" width="100%" />
</p>

该模型训练的时候，评估使用的是`greedy_search`，因此日志中的结果一般不怎么好。`rouge2`只有`20.3577`。
不过后来我对这个结果重新使用了`beam_search`进行评估，然后结果很好。
使用paddlenlp的预测结果。`rouge2`有`21.0125`，达到论文效果。
使用huggingface的预测结果。`rouge2`有`20.9931`，达到论文效果。


#### 使用了paddlenlp进行预测评估(达到论文效果)
```bash
# 转换权重（记得先从百度云把预训练好的结果下载，然后放进step-25000-aistudio-20.35中）
python eval_paddle.py --model_name_or_path ./step-25000-aistudio-20.35 --length_penalty 1.0 --num_beams 4 --eval_batch_size 16 --max_source_length 1024 --max_target_length 512 --evaluate_file ./caches/cnndailymail/cnn_dailymail_dev.json 
```
<p align="center">
    <img src="figure/1x3090-paddle-resut.png" width="100%" />
</p>


#### 使用了huggingface进行预测评估(达到论文效果)
这里使用`4 * RTX3090`，将训练好的权重转化为了huggingface版本，然后预测的结果。
```bash
# 切换路径
cd TASK/CNNDM/eval_huggingface
# 转换权重（记得先从百度云把预训练好的结果下载，然后放进../step-25000-aistudio-20.35中）
python convert_paddle2pytorch.py 
# 配置多卡的环境。（照着对话进行设置，我设置了4卡）
accelerate config
# 运行评估（差不多评估15分钟）
accelerate launch eval_huggingface.py --model_name_or_path ../step-25000-aistudio-20.35 --length_penalty 1.0 --num_beams 4 --per_device_eval_batch_size 16 --max_source_length 1024 --max_target_length 512 --evaluate_file ../caches/cnndailymail/cnn_dailymail_dev.json 
```
<p align="center">
    <img src="figure/4x3090-eval.jpg" width="100%" />
</p>
<p align="center">
    <img src="figure/4x3090-result.jpg" width="100%" />
</p>

由于我将模型预测的摘要结果保存了，因此我们可以直接打开`TASK/CNNDM/eval_huggingface/验证结果.ipynb`，运行全部就可以得到测试结果。


##### 模型链接

链接：https://pan.baidu.com/s/10W3GtWAliUU7euRDs3mp6Q 
提取码：hgx9


# Reference

```bibtex
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
```
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
