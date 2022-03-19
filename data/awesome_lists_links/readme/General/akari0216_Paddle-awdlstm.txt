# Paddle-awdlstm

论文地址：[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146v5.pdf)

**概述**：
归纳迁移学习已经极大地影响了计算机视觉，但现有的NLP方法仍然需要特定任务的修改和从零开始的训练。我们提出了通用语言模型微调(Universal Language Model Fine-tuning, ULMFiT)，这是一种有效的迁移学习方法，可以应用于自然语言处理中的任何任务，并介绍了语言模型微调的关键技术。在6个文本分类任务上，我们的方法显著优于目前最先进的方法，在大多数数据集上降低了18-24%的错误。

## 模型概述
<p align="center">
    <img src="images/model.png" width="100%" />
</p>

## 原论文效果
<p align="center">
    <img src="images/ag_news.png" width="100%" />
</p>

## 开始
### 1.预训练权重下载
链接：https://pan.baidu.com/s/1wTaAFGFKKlHoFI92Pf4sIw 
提取码：wtrq

Wikitext-103是超过 1 亿个语句的数据合集，全部从维基百科的 Good 与 Featured 文章中提炼出来。广泛用于语言建模，当中 包括 fastai 库和 ULMFiT 算法中经常用到的预训练模型。该权重为基于此数据集上预训练后得到的
权重包含了已转换的fwd和bwd两个权重


### 2.模型微调
AG News Dataset 拥有超过 100 万篇新闻文章，其中包含 496,835 条 AG 新闻语料库中超过 2000 个新闻源的文章，该数据集仅采用了标题和描述字段，每种类别均拥有 30,000 个训练样本和 1900 个测试样本。

本次微调过程均在AIStudio上进行，使用GPU模式进行
步骤：
1.进入bash执行
pip install paddlepaddle
pip install paddlenlp
pip install spacy
pip install --upgrade numpy

2.下载预训练权重，将converted_fwd.pdparams和converted_bwd.pdparams放入根目录

3.进入bash，执行命令sh run_pretrain.sh
在每一个阶段会生成当前最佳acc的权重并且保存，作为下一个阶段微调的预加载权重
所有的language model finetune和text classifier finetune的日志都记录在log文件夹下

### 3.数据集验证
进入bash执行
python merge_preds.py
分别对converted_fwd.pdparams和converted_bwd.pdparams的微调结果生成对应的结果文件
然后再执行
python create_final_preds.py
对两个结果文件进行融合，得到最终的预测

由于时间关系，只有对converted_fwd.pdparams微调后的预测结果，
该权重对测试集的预测acc如下：

## fwd预测效果
<p align="center">
    <img src="images/fwd_res.png" width="100%" />
</p>

而若在与converted_bwd.pdparams微调后融合的结果下，理论上能提升0.5-0.7pp
## 论文理论效果
<p align="center">
    <img src="images/combine.png" width="100%" />
</p>


可使用已微调的forward权重和backward权重来查看效果。
已微调权重链接下载：
链接：https://pan.baidu.com/s/18UB_irYi6yRZJMsZKUwC9w 
提取码：0wyy 

将已微调权重放在根目录后，执行命令sh run_final_pred.sh
## 最终融合效果
<p align="center">
    <img src="images/merge_final.png" width="100%" />
</p>

在一定误差范围内达到论文理论效果