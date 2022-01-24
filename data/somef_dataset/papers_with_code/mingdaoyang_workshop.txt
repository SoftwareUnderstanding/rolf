## 项目一：问答摘要与推理

###   1.项目简介

 项目源于[百度PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理](http://ai.baidu.com/forum/topic/show/864802)  

###   2.数据集简介

​数据产生的场景是这样的，一些车主在自己的爱车遇到问题时，在汽车大师App上发起提问，专业技师会根据问题（Problem）和用户进行一段对话（Conversation），从而帮助用户解决问题，最后技师根据问题和对话生成一个报告(Report)
  ![image](https://github.com/mingdaoyang/workshop/blob/master/imges/20200430153435.jpg)

###  3.相关Paper
      * Week1：
      1)Word2Vec Tutorial - The Skip-Gram Model，http://mccormickml.com/2016/04/19/word2vec-tutorial-the-s)
      2)Efficient Estimation of Word Representations in Vector Space (https://arxiv.org/pdf/1301.3781.pdf)
      * Week2：
      xin rong, word2vec Parameter Learning Explained https://arxiv.org/abs/1411.2738)(公式推导的很详细很全面)
      * Week3：
      1)Sequence Modeling: Recurrentand Recursive Nets http://www.deeplearningbook.org/contents/rnn.html
      2)Attention and Augmented Recurrent Neural Networks https://distill.pub/2016/augmented-rnns/
      

###  4.数据预处理与Word2Vec词向量训练

- 主要使用了开源的**jieba**分词和**pandas**包（后面词向量训练用到了gensim）；
- 简单的清洗，主要去除一些无效字符，stopwords只包含了英文停用词；
- 将数据集划分为四个部分，即train_x,train_y,test_x,test_y(需要预测生成，初始为空)，合并前三个部分后，在生成一个word+index的vocab；
- 使用gensim训练词向量，为提高相似词的查询效率，先通过gensim的KeyedVectors加载预训练的二进制模型，调用model.vocab.key()，按照{index：word}写入到   一个JSON文件，然后使用Annoy的AnnoyIndex，具体代码如下：
    ```Python
    from annoy import AnnoyIndex
    .......
    wv_index = AnnoyIndex(256, metric='angular') # ?
    i = 0
    for key in wv_model.vocab.keys():
        v = wv_model[key]
        wv_index.add_item(i, v)
        i += 1
    wv_index.build(10)  # 10 trees
    ```
    最后生成{index->vector}的一个dict，这样就可以作为S2S等后续需要学习模型的embedding层。
    
###  5.Seq2Seq模型的使用()
