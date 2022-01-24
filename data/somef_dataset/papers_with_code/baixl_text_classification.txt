[TOC]
### 1、项目描述
基于pytorch实现的中文文本分类
### Requirement
```
python  3.7
pytorch 1.1
tensorboardx
numpy
torchvision
torchtext
pytorch-pretrained-BERT： pip install pytorch-pretrained-bert
```

### 2、数据
```
数据集： 搜狗实验室新闻数据集
embedding: https://github.com/Embedding/Chinese-Word-Vectors
这里使用以字为单位的300维 词向量：（Sogou News 搜狗新闻）
```
[搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)

### 3、运行示例
```
python main.py --model FastText / TextCNN / TextRNN / TextRCNN / TextRNN_Att

Bert:
python main_bert.py --model Bert
```

效果：RTX2070上训练数据

| model | precision | recall |  f1-score | 训练时间|
|------ | ------    | ------ | ------    | ------ |
| FastText|  0.9238 |0.9237 |0.9236 |0:06:02 |
| TextCNN |0.9110 |0.9108 |0.9108 |  0:03:11|
| TextRNN | 0.9057 |0.9045| 0.9047|0:01:51|
| TextRCNN |0.9135|0.9129 |0.9129|0:03:10|
| TextRNN_Att |0.9205|0.9204|0.9203|0:05:53|
| Transformer |0.8955|0.8944| 0.8943|0:10:27|
| Bert |0.9412|0.9411|0.9410 | 0:45:13|

### 4、model细节
#### 4.1、fasttext
论文： Bag of Tricks for Efficient Text Classification

结构：
```
embedding + bi_gram + tri_gram  --> 拼接  -->fc (fc1 --> fc2) --> softmax
```
![fasttext](./images/fastText.jpeg)

>https://www.jianshu.com/p/48dd04212f48
> 对于普通 word2vec，输入层就是一个词向量的查找表，所以它的大小为 nwords 行，dim 列（dim 为词向量的长度），但是 fastText 用了 word n-gram 作为输入，所以输入矩阵的大小为 (nwords + ngram 种类) * dim。代码中，所有 word n-gram 都被 hash 到固定数目的 bucket 中，所以输入矩阵的大小为 (nwords + bucket 个数) * dim.

>词向量的个数是单词的数量加上桶的个数。多个ngram或是字符ngram会在一个桶中共享一个向量


作者：骑鲸公子_
链接：https://www.jianshu.com/p/48dd04212f48
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#### 4.2、textCNN
论文 ：Convolutional Neural Networks for Sentence Classification

结构：
```
embedding --> conv(1维) --> max pooling -->full connected layer --> soft_max
```

![textCNN](./images/TextCNN.jpeg)

#### 4.3、textRNN
论文 ：Recurrent Neural Network for Text Classification with Multi-Task Learning

结构：
```
embedding --> bi-lstm --> 拼接 --> average --> softmax
```

#### 4.4、RCNN
论文：Recurrent Convolutional Neural Network for Text Classification

结构：

```
recurrent structure (convolutional layer)  --> max pooling  --> fc --> softmax

RNN 融合了左右上下文信息：
[left_side_context_vector,current_word_embedding,right_side_context_vecotor]
```
![textRCNN](./images/RCNN.jpeg)

#### 4.5、HAN
论文：Hierarchical Attention Networks for Document Classification

结构：
```
1、embedding
2、word encoder
3、word attention
4、sentence encoder
5、sentence attention
6、fc 
7、softmax
```
![HAN](./images/HAN.jpeg)
参考：[Hierarchical Attention Network for Document Classification阅读笔记](https://blog.csdn.net/liuchonge/article/details/73610734)

#### 4.6、Att-BiLSTM

论文：Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

论文解读：https://blog.csdn.net/qq_36426650/article/details/88207917

github: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

结构：
```
输入层 -->embedding --> BiLSTM --> attention层 --> 输出层
```
![Att-BiLSTM](./images/Att-BiLSTM.PNG)



#### 4.7、Transformer
论文：Attention Is All You Need

[1、Attention机制详解（二）——Self-Attention与Transformer](https://zhuanlan.zhihu.com/p/47282410)

[2、Transformer 模型的 PyTorch 实现](https://juejin.im/post/5b9f1af0e51d450e425eb32d)

[3、详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)
[《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)

[4、nlp Transformer详解](https://zhuanlan.zhihu.com/p/44121378)

结构：

```
1、embedding
2、positional_Encoding
3、Encoder
	3.1、Multi_Head_Attention
		Scaled_Dot_Product_Attention
	3.2、Position_wise_Feed_Forward

transforer 是经典的Encoder-Decoder架构
```

![transformer](./images/Transformer.png)

![transformer](./images/sccaled-dot-product-attention.jpeg)

### 5、Bert
这里讲bert单独拎出来。
```
pip install pytorch-transformers

https://github.com/huggingface/pytorch-transformers
```
[1、BERT详解](https://zhuanlan.zhihu.com/p/48612853)

更多的预训练model 看github：  

https://github.com/huggingface/pytorch-transformers

![预训练](./images/pytorch_transformers.png)

使用bert时，需要深入理解nlp预训练的发展。  

1. 预训练需要学习一个语言模型，对文本进行抽象
2. 语言模型的表征能力，需要语境和语义信息(动态、多义)

预训练的发展：

```
1、word2vec (glove、 fasttext) 
2、Elmo(Lstm 双向) 
3、GPT(Transformer 单向) 
4、Bert(双向Transformer + mask)
5、ERINE 
6、XLNet(Transformer-XL， 归纳了AR:AutoRegression AE:AutoEncoding 两种语言模型的长处)

ELMO: 双向LSTM，输出3个向量： 预训练词向量、句法特征、语义特征，下游使用时可以直接使用三个向量做拼接使用
GPT： 首次使用Transformer代替 RNN作为特征抽取器(GPT中只是使用了单向的特征提取)。 训练输出可以随着下游 fine-tune

Bert：
    1、使用Transformer作为特征抽取器(双向)
    2、使用Mask Language Model(MLM)和 Next Sentence Prediction(NSP)的多任务目标
    15%的word被随机mask掉：训练中对这15%的词，以80%的概率直接mask， 
    10%概率替换成其他任意单词，10%概率保留原来的token。 这样迫使Transformer学习到word的上下文信息。
    (原因：如果某个单词被100%概率mask，在fine-tune时就看不到这个词，加入随机token， 可以让Transformer保持对每个输入token的分布式表征，否则在self-attention中，上一层就能记住这个呗mask的token， 
    一个单词最终被随机替换的概率是 15% * 10% =1.5%  负面影响可以忽略不计）

XLNET：TODO

```

### 6、参考资料&论文

1、文本分类model：
- FastText: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf)
- TextCNN: [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181) 
- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- char_TextCNN: [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) 
- TextRNN: [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/pdf/1605.05101.pdf)
- HAN(textRNN + attention): [Hierarchical Attention Networks for Document Classification](http://link.zhihu.com/?target=https%3A//www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
- Att-BiLSTM:[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034)
- TextRCNN(TextRNN+CNN): [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)
- 知乎--[用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践](https://zhuanlan.zhihu.com/p/25928551) 

2、预训练

- ELMo--[Deep contextualized word representations](http://www.aclweb.org/anthology/N18-1202)
- ULMFiT--[Universal Language Model Fine-tuning for Text Classification](http://aclweb.org/anthology/P18-1031)
- GPT--[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Transformer: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

- BERT--[Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- transformer-XL:
- XLNet [XLNet: Generalized Autoregressive Pretraining
for Language Understanding](https://www.researchgate.net/publication/333892322_XLNet_Generalized_Autoregressive_Pretraining_for_Language_Understanding)
- XLNet介绍 [20项任务全面碾压BERT，CMU全新XLNet预训练模型屠榜（已开源）](https://www.jiqizhixin.com/articles/2019-06-20-9)
- ERINE 2.0 https://github.com/PaddlePaddle/ERNIE/blob/develop/README.zh.md

注意力：

- [Attention机制详解（二）——Self-Attention与Transformer](https://zhuanlan.zhihu.com/p/47282410)
- [深度学习中的注意力模型（2017版）](https://zhuanlan.zhihu.com/p/37601161)

- [Transformer 模型的 PyTorch 实现](https://juejin.im/post/5b9f1af0e51d450e425eb32d)
- [详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)
- [《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)
- [BERT详解](https://zhuanlan.zhihu.com/p/48612853)
- [从word Embedding 到BERT-自然语言处理中的于训练技术发展史-by 张俊林](https://zhuanlan.zhihu.com/p/49271699)

### 7、github

*  [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
tensorflow实现的字符级textCnn RNN
* [text_classification-- brightmart](https://github.com/brightmart/text_classification)
tensorflow实现的各种版本文本分类model，知乎看山杯比赛的数据集
*  https://github.com/Magic-Bubble/Zhihu 
基于pytorch实现的实现的各类文本分类model
*  https://github.com/649453932/Chinese-Text-Classification-Pytorch  
中文pytorch实现的文本分类
*  https://github.com/songyingxin/TextClassification-Pytorch

词向量下载

* pytorch加载bert ERINE等： https://github.com/huggingface/pytorch-transformers
* ERINE 转换未pythorch 加载的bert格式：https://github.com/nghuyong/ERNIE-Pytorch
* bert：https://github.com/google-research/bert
