# 使用PaddlePaddle复现论文：ConvBERT: Improving BERT with Span-based Dynamic Convolution

## ConvBERT

[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)

**摘要：**
像BERT及其变体这样的预训练语言模型最近在各种自然语言理解任务中取得了令人印象深刻的表现。然而，BERT严重依赖全局自注意力块，因此需要大量内存占用和计算成本。
虽然它的所有注意力头从全局角度查询整个输入序列以生成注意力图，但我们观察到一些头只需要学习局部依赖，这意味着存在计算冗余。
因此，我们提出了一种新颖的基于跨度的动态卷积来代替这些自注意力头，以直接对局部依赖性进行建模。新的卷积头与其余的自注意力头一起形成了一个新的混合注意力块，在全局和局部上下文学习中都更有效。
我们为 BERT 配备了这种混合注意力设计并构建了一个ConvBERT模型。实验表明，ConvBERT 在各种下游任务中明显优于BERT及其变体，具有更低的训练成本和更少的模型参数。
值得注意的是，ConvBERT-base 模型达到86.4GLUE分数，比ELECTRA-base高0.7，同时使用不到1/4的训练成本。

本项目是 ConvBert 在 Paddle 2.x上的开源实现。

## 原论文效果
<p align="center">
    <img src="figure/qnli.jpg" width="100%" />
</p>
<p align="center">
    <img src="figure/squad.jpg" width="100%" />
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
git clone https://github.com/JunnYu/paddle_convbert
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
运行`python compare.py`，对比huggingface与paddle之间的精度，我们可以发现精度的平均误差在10^-7量级，最大误差在10^-6量级。
```python
python compare.py
# huggingface YituTech/conv-bert-small vs paddle convbert-small
# mean difference: tensor(4.6980e-07)
# max difference: tensor(2.8610e-06)
# huggingface YituTech/conv-bert-medium-small vs paddle convbert-medium-small
# mean difference: tensor(3.4326e-07)
# max difference: tensor(2.8014e-06)
# huggingface YituTech/conv-bert-base vs paddle convbert-base
# mean difference: tensor(4.5306e-07)
# max difference: tensor(8.1062e-06)
```

### （二）模型预训练

**特别注意**：预训练模型如果想要达到较好的效果，需要训练几乎全量的Book Corpus数据 和 Wikipedia Corpus数据，原始文本接近20G，建议用GPU进行预训练，最好4片GPU以上。如果资源较少，Paddle提供已经预训练好的模型进行Fine-tuning，可以直接跳转到下面：运行Fine-tuning-使用Paddle提供的预训练模型运行 Fine-tuning。

单机单卡下进行训练：

```shell
export CUDA_VISIBLE_DEVICES="0"
export DATA_DIR=./BookCorpus/

python -m paddle.distributed.launch --gpus "0" run_pretrain.py \
    --model_type convbert \
    --model_name_or_path convbert-small \
    --input_dir $DATA_DIR \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 4 \
    --logging_steps 100 \
    --save_steps 10000 \
    --max_steps -1 \
    --device gpu
```
其中参数释义如下：
- `model_type` 表示模型类型，默认为ConvBERT模型。
- `model_name_or_path` 如果配置1个名字，则表示预训练模型的规模，当前支持的名字为：convbert-small、convbert-medium-small、convbert-base。如果配置1个路径，则表示按照路径中的模型规模进行训练，这时需配置 --init_from_ckpt 参数一起使用，一般用于断点恢复训练场景。
- `input_dir` 表示输入数据的目录，该目录下需要有1个train.data纯英文文本文件，utf-8编码。
- `output_dir` 表示将要保存预训练模型的目录。
- `train_batch_size` 表示 每次迭代**每张卡**上的样本数目。此例子train_batch_size=64 运行时大致需要单卡12G显存，如果实际GPU显存小于12G或大大多于12G，可适当调小/调大此配置。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `weight_decay` 表示每次迭代中参数缩小的比例，该值乘以学习率为真正缩小的比例。
- `adam_epsilon` 表示adam优化器中的epsilon值。
- `warmup_steps` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存间隔。
- `max_steps` 如果配置且大于0，表示预训练最多执行的迭代数量；如果不配置或配置小于0，则根据输入数据量、train_batch_size和num_train_epochs来确定预训练迭代数量
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

另外还有一些额外参数不在如上命令中：
- `use_amp` 表示是否开启混合精度(float16)进行训练，默认不开启。如果在命令中加上了--use_amp，则会开启。
- `init_from_ckpt` 表示是否从某个checkpoint继续训练（断点恢复训练），默认不开启。如果在命令中加上了--init_from_ckpt，且 --model_name_or_path 配置的是路径，则会开启从某个checkpoint继续训练。例如下面的命令从第40000步的checkpoint继续训练：

### （三）下游任务微调

#### 1、GLUE
以QNLI数据集为例（对于其他GLUE任务，请参考logs/GLUE/对应task_name/args.json，该json有详细参数配置）

##### （1）模型微调：
```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在glue文件夹
cd glue
# 运行训练
python -m paddle.distributed.launch --gpus "0" run_glue.py \
    --model_type convbert \
    --model_name_or_path convbert-base \
    --task_name qnli \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --scheduler_type cosine \
    --layer_lr_decay 0.8 \
    --weight_decay 0.01 \
    --warmup_proportion 0.1 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --seed 42 \
    --output_dir qnli/ \
    --device gpu
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA、ERNIE、CONVBERT模型。
- `model_name_or_path` 模型名称或者路径，其中convbert模型当前仅支持convbert-small、convbert-medium-small、convbert-base几种规格。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE、 WNLI。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `scheduler_type` scheduler类型，可选linear和cosine
- `layer_lr_decay` 层学习率衰减，参考原TF代码。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

**模型链接**(这个链接包含所有GLUE任务微调后的权重)

链接：https://pan.baidu.com/s/1-Pk3n7jJj3IjaXZYR_J-xw 
提取码：zuif

##### （2）模型预测：
```bash
# 确保处在glue文件夹
cd glue
# 运行预测，请指定模型权重文件夹
python run_predict.py --task_name qnli  --ckpt_path qnli/best-qnli_ft_model_6300.pdparams
# 完成后可以压缩template文件夹，然后提交到GLUE
```
##### （3）压缩template文件夹为zip文件，然后提交到[GLUE排行榜](https://gluebenchmark.com/leaderboard)：
###### GLUE排行榜结果：
<p align="center">
    <img src="figure/glue.png" width="100%" />
</p>

###### GLUE开发集结果：
| Model                          | cola  | sst-2  | mrpc        | sts-b             | qqp         | mnli       | qnli | rte   | wnli  |
|--------------------------------|-------|-------|-------------|------------------|-------------|-------------|------|-------|-------|
|                                | mcc   | acc   | acc/f1      | pearson/spearman | acc/f1      | acc(m/mm)      | acc  | acc   | acc   |
| ConvBERT\-base\-Paddle\(Mine\) | 69\.2 | 94\.6 | 90\.4/93\.1 | 90\.8/90\.7      | 92\.0/89\.4 | 88\.1/87\.9 | 93   | 83\.4 | 56\.3 |



###### GLUE测试集结果对比（论文中的结果）：
| Model                          | cola  | sst-2  | mrpc  | sts-b  | qqp | mnli-m | qnli  | rte   | avg      |
|--------------------------------|-------|-------|-------|-------|-----|-------|-------|-------|----------|
| ConvBERT\-base\-YiTu           | **67\.8** | **95\.7** | 88\.3 | **89\.7** | **90.0**  | **88\.3** | 93\.2 | 77\.9 | **86\.3625** |
| ConvBERT\-base\-Paddle\(Mine\) | 66\.3 | 95\.4 | **88\.6** | 89\.2 | **90.0**  | 88\.2 | **93\.3** | **78\.2** | 86\.15   |

#### 2、SQuAD v1.1

使用Paddle提供的预训练模型运行SQuAD v1.1数据集的Fine-tuning

```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在squad1.1文件夹
cd squad1.1
# 开始训练
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type convbert \
    --model_name_or_path convbert-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --scheduler_type linear \
    --layer_lr_decay 0.8 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir squad/ \
    --device gpu \
    --do_train \
    --seed 42 \
    --do_predict
```

训练过程中模型会自动对结果进行评估，其中最好的结果如下所示：（详细训练可查看logs文件夹）

```python
# 注意：paddlenlp的tokenizer的offset mapping有误,存在BUG!!!!!!，该结果有点低！！！！！。
# gloabl step = 9500
{
  "exact": 84.46546830652791,
  "f1": 90.9998989286406,
  "total": 10570,
  "HasAns_exact": 84.46546830652791,
  "HasAns_f1": 90.9998989286406,
  "HasAns_total": 10570
}

# 这是使用 python evaluate.py运行得到的结果！！！这个结果才是正确的！！！！！
{
  "exact": 84.82497634815516,
  "f1": 91.29326590096191,
  "total": 10570,
  "HasAns_exact": 84.82497634815516,
  "HasAns_f1": 91.29326590096191,
  "HasAns_total": 10570
}
```

##### 模型链接

链接：https://pan.baidu.com/s/1TJnc976TM694TEs9HgF66w 
提取码：bt5m

#### 3、SQuAD v2.0
对于 SQuAD v2.0,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
# 确保处在squad2.0文件夹
cd squad2.0
# 开始训练
python -m paddle.distributed.launch --gpus "0" run_squad.py \
    --model_type convbert \
    --model_name_or_path convbert-base \
    --max_seq_length 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --scheduler_type linear \
    --layer_lr_decay 0.8 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir squad/ \
    --device gpu \
    --do_train \
    --do_predict \
    --seed 42 \
    --version_2_with_negative
```

* `version_2_with_negative`: 使用squad2.0数据集和评价指标的标志。

训练过程中模型会自动对结果进行评估，其中最好的结果如下所示：（详细训练可查看logs文件夹）

```python
# 注意：paddlenlp的tokenizer的offset mapping有误,存在BUG!!!!!!，该结果有点低！！！！！。
# global step = 14000
{
  "exact": 80.94837025183189,
  "f1": 83.96743280565248,
  "total": 11873,
  "HasAns_exact": 74.67948717948718,
  "HasAns_f1": 80.72627019256272,
  "HasAns_total": 5928,
  "NoAns_exact": 87.19932716568545,
  "NoAns_f1": 87.19932716568545,
  "NoAns_total": 5945,
  "best_exact": 80.98206013644403,
  "best_exact_thresh": -0.5712471008300781,
  "best_f1": 83.96743280565195,
  "best_f1_thresh": -0.0010485649108886719
}

# 这是使用 python evaluate.py运行得到的结果！！！这个结果才是正确的！！！！！
{
  "exact": 80.9062578960667,
  "f1": 83.90042392251083,
  "total": 11873,
  "HasAns_exact": 75.10121457489879,
  "HasAns_f1": 81.09813313629743,
  "HasAns_total": 5928,
  "NoAns_exact": 86.69470142977292,
  "NoAns_f1": 86.69470142977292,
  "NoAns_total": 5945,
  "best_exact": 81.23473427103512,
  "best_exact_thresh": -0.7430887222290039,
  "best_f1": 84.1231432489553,
  "best_f1_thresh": -0.7430887222290039
}
```

##### 模型链接

链接：https://pan.baidu.com/s/1s2W5BgBa0eqLywjDa6Kv8A 
提取码：u0rn 

# Reference

```bibtex
@article{Jiang2020ConvBERT,
  title={ConvBERT: Improving BERT with Span-based Dynamic Convolution},
  author={Zi-Hang Jiang and Weihao Yu and Daquan Zhou and Y. Chen and Jiashi Feng and S. Yan},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.02496}
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