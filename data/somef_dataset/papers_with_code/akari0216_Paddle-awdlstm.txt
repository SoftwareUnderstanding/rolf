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
链接：https://pan.baidu.com/s/1eA0XC8T3g6-bXvg20WyHnQ 
提取码：q136

Wikitext-103是超过 1 亿个语句的数据合集，全部从维基百科的 Good 与 Featured 文章中提炼出来。广泛用于语言建模，当中 包括 fastai 库和 ULMFiT 算法中经常用到的预训练模型。该权重为基于此数据集上预训练后得到的
权重包含了fwd和bwd两个权重


### 2.模型微调

### 3.数据集验证
AG News Dataset 拥有超过 100 万篇新闻文章，其中包含 496,835 条 AG 新闻语料库中超过 2000 个新闻源的文章，该数据集仅采用了标题和描述字段，每种类别均拥有 30,000 个训练样本和 1900 个测试样本。

步骤：
1.进入bash执行
pip install paddlepaddle
pip install paddlenlp
pip install spacy
pip install --upgrade numpy

2.下载预训练权重，将wt103-fwd2.pdparams放入config文件夹

3.去到各py文件手动调整路径

4.进入bash 分别执行
python lm_data.py
python cls_data.py
python test_data.py