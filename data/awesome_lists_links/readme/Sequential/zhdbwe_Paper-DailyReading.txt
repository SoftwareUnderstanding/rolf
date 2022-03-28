# Paper-DailyReading

Dataset: 

单文档摘要(single document)
1. [CNN/DM](https://github.com/abisee/cnn-dailymail) 
    
    这个数据集是分别从CNN和DailyMail采集的数据，有匿名版本和非匿名版本，这里是非匿名版本。包含了287,227 训练集 13,368 验证集 11,490 测试集。
2. [GigaWord](https://github.com/facebookarchive/NAMAS)

    这个数据集中的摘要是指标题(一般只有一句话),包含了3.8M 训练集 189K 验证集 和 1951 测试集
3. [XSum](https://github.com/EdinburghNLP/XSum)

    新提出的专注于生成式摘要任务的数据集， 包含204,045 训练集 11,332 验证集 11,334 测试集


多文档摘要(multi document)
1. [DUC](https://duc.nist.gov/)
2. [WikiSum](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/wikisum)



|Models|RG-1|RG-2|RG-L|
|------|----|----|----|
|ETADS|41.75|19.01|38.89| 

|Models|RG-1|RG-2|RG-L|
|------|-----|----|----|
|BART|44.16|21.28|40.90|

2019-11-07

[Improving Abstractive Document Summarization with Salient Information Modeling](https://www.aclweb.org/anthology/P19-1205.pdf) ACL2019


这篇论文针对长文本编码和有效信息抽取的问题，提出了两种机制去解决。都是在注意力的基础上进行修改的。

1. focus attention 用于解决长文本编码问题，考虑到局部的信息，在query和key的运算之后加入高斯噪声，使其在远离注意力中心的单词权重更小一些。
2. selection network 用于解决有效信息抽取的问题，用key和query计算出当前单词的有效信息打分，然后将打分乘上注意力的权重

2019-11-08

[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) arxiv

这是一篇关于预训练的论文，和之前的预训练任务不同，本篇论文专注于生成任务。

创新点主要有两点：
1. 和其他预训练模型不同，本模型同时使用了encoder(双向用于编码信息)和decoder(单向用于生成任务),好像和原始的transformer也没有什么区别
2. 提出了不同的掩码和变换方式,包括了Token masking ,token deletion, text infilling, sentence permutation, document rotation

2019-11-18

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf) arxiv

本论文对原始Transformer模型只能编码固定长度的数据进行改进(Vaswani Transformer -> Universal Transformer -> Transformer-XL)

1. 使用RNN的思想，将前面的状态作为历史信息输入到下一个时间步，和RNN不同的地方在于他不是在同一层输入历史信息，而是利用了上一层的历史信息
2. 使用了相对位置编码，防止绝对位置编码时，两个不同的Block的同一位置不能区分。

2019-11.25

[Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection](https://openreview.net/forum?id=Hye87grYDH)

本文主要是让注意力更集中，实现方法，直接在每次注意力之后 ![](http://latex.codecogs.com/gif.latex?\\frac{QK^T}{\sqrt{d}}) 注意力权重只取top-k, k为超参数。

2019-11-27

[Searching for Effective Neural Extractive Summarization: What Works and What’s Next](https://arxiv.org/pdf/1907.03491.pdf)

本论文主要对抽取式摘要任务的一些框架和各种模块进行了比较，
1. Encoder(Lstm效果会好一些但是容易过拟合, Transformer模型更加鲁棒)
2. Decoder(自回归,auto-regressive,这里使用的是pointer的decoder 和非自回归non auto-regressive,这里使用的是序列标注方法)一般来说自回归的方法效果较好
3. Position information(CNN数据集非常依赖位置信息,文中多个实验得出相同的额结论(sentence shuffle, distangling test etc.))
4. 引入外部知识(无监督bert..和有监督(在某些额外的数据集上预训练然后迁移到本数据及)的预训练),一般无监督的效果较好，有监督的有domain shift problem
5. 学习方法(预训练,强化学习,监督学习是互补的, 可以在已有的方法上加入新的学习方法依然可以得到提高)

2019-12-04

[MeanSum : A Neural Model for Unsupervised Multi-Document Abstractive Summarization](http://proceedings.mlr.press/v97/chu19b/chu19b.pdf)

本文的任务是多文档摘要，作者所提出一种新的利用无监督的方式生成摘要。

![模型架构](https://github.com/zhdbwe/Paper-DailyReading/raw/master/pic/mean_sum.png)

整个模型分为两个大块，
1. 首先利用自编码器学习文档的编码(自编码器可以是预训练的)
2. 摘要生成部分，简单的多篇文档的编码状态做一个平均，然后利用自编码器中的解码器生成摘要，再对摘要进行编码并且和所有文档的编码计算相似度作为损失。(值得注意的生成的摘要是离散状态，不能直接进行梯度反向传播,作者利用了Straight Through Gumbel-Softmax trick)
