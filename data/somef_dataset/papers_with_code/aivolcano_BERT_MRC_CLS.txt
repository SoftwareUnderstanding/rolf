### 任务目标
* 囊括NLP领域的最新paper中提到的模型, 作者开发该框架，对于新的paper，只需要看model.py就可以融入该框架，并且可以考虑迁移到推荐系统中使用
* 推荐系统中最重要的特征是ID类特征，它组成的用户行为句子依靠NLP技术进行特征提取，NLP提取特征的技术路线为推荐系统提供了可参考的技术路线，2021年推荐系统领域内的模型逐步向预训练BERT靠拢，以求能提取更有效的提取ID组成的用户行为句子特征
* 该框架已经包括语义匹配、文本分类、阅读理解3个框架，未来继续增加文本生产和命名实体识别2部分，这样，NLP的框架内容基本就已经囊括了

### 核心技术
BERT隐藏层加权融合，召回段落、召回关键句 等等

### 特征工程
* 召回技术
    * TextRank长文本召回关键句，
    * BM25算法：跨文档召回top-k个含有正确答案的段落

* Pooling：BERT 12层隐藏层的向量进行加权

Ganesh Jawahar等人[3]证明BERT每一层对文本的理解都不同，因此将BERT的十二层transformer生成的表示赋予一个权重,(下文称为BERT动态融合)，初始化公式为：

α_i = Dense_(unit=1) (represent_i)

ouput = Dense_(unit=512) （∑_(i=1)^nα_i · represent_i）

BERT的动态融合作为模型embedding的成绩会优于BERT最后一层向量作为模型的embedding，且收敛速度更快，epochs=3-4个就可以了

![image](https://user-images.githubusercontent.com/68730894/115149174-72b57d00-a095-11eb-9b2a-68f128c542b2.png)

权重学习易造成高时空复杂度，我们还可以使用SumPooling、MeanPooling、MaxPooling等方法进行融合，选择层数偏后面几层。


### 模型内部结构 Pooling
* 使用残差网络解决BERT12个隐藏层喂给BiLSTM时，模型发生退化问题
我们将BERT的12个隐藏层喂给BiLSTM，即：BERT -> BiLSTM -> Average Pooling的技术路线。实际上，作者在训练的过程中发现，增加LSTM之后，模型发生退化问题。因此我们开发了残差网络，让模型自己选择要不要跳过BiLSTM。

![image](https://user-images.githubusercontent.com/68730894/115556000-be109b00-a2e2-11eb-91a6-929d151f4e1c.png)


作者还提供更为丰富的模型替换LSTM，比如：GRU、Transformer、CNN、RTransformer(4选1) 等结构。由于使用残差网络，残差网络是加法，时空复杂度为O(1)，参数量不变。

![image](https://user-images.githubusercontent.com/68730894/115149184-88c33d80-a095-11eb-94be-fdefcb3f6d6d.png)

出现退化原因可能是：BERT的12层向量融合完成很好的提取了特征，这种情况复杂的模型反而效果会减弱。这在推荐系统中很常见，特征工程之后用个逻辑回归LR就能解决问题，可能对于LR来说，它只需要发挥自己的记忆能力，把特征工程整理出来的情况都记录在自己的评分卡中。

* 开发不受维度限制的残差模型
顺着这一逻辑往下走，作者开发了不受维度限制的残差模块，希望解决维度不一致影响残差网络使用率低的问题。笔者认为在推荐系统和自然语言处理的深度学习模型都可以使用，包括特征提取层（Feature extrusion）和 全连接层（MLP）。它可以作为模型内部特征前后复用的一种策略。此外，残差网络时空复杂度变化小，且能保证效果达到最优时不会变差。

LSTM的残差连接
```python
class ResidualWrapper4RNN(nn.Module):
    def __init__(self, model):
        super().__init__() # super(ResidualWrapper, self).__init__()
        self.model = model
    def forward(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)[0]  # params.model_type=='bigru' or 'bilstm'
        return inputs + delta

self.bilstm = ResidualWrapper4RNN(nn.Sequential(
            nn.BiLSTM(config.hidden_size, params.lstm_hidden,
                             num_layers=params.num_layers, dropout=params.drop_prob, batch_first=True, bidirectional=True)
                             ))

result = self.bilstm(bert_ouput)
```

原理是：回到ResNet的核心，非线性激活函数的存在导致特征变化不可逆，因此造成模型退化的根本原因是非线性激活函数。因此F(x)= f(x) + x 可以理解为f(x)为非线性特征，x为线性特征。

![image](https://user-images.githubusercontent.com/68730894/115149195-9678c300-a095-11eb-8a53-e005612c6e7e.png)

作者开发的不受维度限制的残差网络数学公式是： 

F(x)= f(x) + wx 

该残差模块不受维度相等的条件限制，w的作用是维度变换，经过w的变换后，特征依然是线性的。

时空复杂度为0的写法：遇到维度不相等，可以直接用`nn.Linear(), tf.keras.layers.Dense()`让维度一致。然后再对位相加即可。

有时空复杂度的写法：向量不对位相加，直接拼接`torch.cat([vector1, vector2],dim=-1), tf.concat([vector1, vector2], axis=-1) tf.keras.layers.concatation()`

![image](https://user-images.githubusercontent.com/68730894/115149220-b0b2a100-a095-11eb-9dea-f38c5089964b.png)

### BERT作为特征提取器可能存在的问题
上述模型中，BERT的参数是不参与模型更新的，虽然我们设置了学习率，但是我们只使用了BERT的输出向量(batch_size, seq_len, embedding_dim)，梯度无法传导到BERT中，原因是BERT 是基于Transformer Encoder完成的，在Transformer中 Decoder 模块的梯度传导可以有交叉信息熵算出来，由于 Decoder 中涉及 Encoder-Decoder Self-Attention，该注意力机制中  q 来自encoder ，k 和 v 来自decoder，随着decoder的更新，decoder中的注意力矩阵完成参数更新，该注意力矩阵中有来自encoder的q，q的更新带动encoder中的注意力矩阵更新，最终实现整个transformer的参数更新。

解决方案：为BERT增加辅助损失函数。

BERT的下游结构我们下接了BiLSTM

### 可改进的点
BERT模型动态融合需要BERT预训练模型已经很完美，因此可以使用我们该任务的语料喂给开源的预训练模型再训练20个epoch。

BERT Fine_tune微调依靠BERT自身已经能很好的提取特征，实际上，使用BERT时，经常遇到loss很小，但是总感觉模型没有学明白的问题，因此，作者考虑为BERT增加有监督学习的loss：
[代码跑通，没有gpu还没跑出结果]BERT是无监督学习方式，该任务是文本分类，因此我们可以为BERT单独增加损失函数，相当于期中考试。具体做法是BERT最后一层的输出（也就是原始BERT）使用LR计算得到pre_label，与true_label计算得到损失


* 修改loss：为bert增加辅助损失函数
```python
# BERT增加损失函数：原始BERT输出直接做分类后计算一次损失（也由于BERT的重要性高于fine_tune部分，其loss权重可以高于fine_tune部分的权重）
ori_pooled_output = self.bert_cls(ori_pooled_output) #(none, 768) -> (none, 10)
bert_cls = F.softmax(ori_pooled_output, dim=-1)
bert_loss = nn.CrossEntropyLoss()(bert_cls, cls_label)

# 或者参考不均衡样本处理方法进行下采样，下采样是通用的思路 使用FocalLoss
class_loss = nn.CrossEntropyLoss()(classifier_logits, cls_label)# weight中设置不均衡的标签
class_loss = 0.8 * bert_loss + 0.2 * class_loss
outputs = class_loss
```

* 为Attention矩阵增加残差，在MultiAttention那里增加残差网络

![image](https://user-images.githubusercontent.com/68730894/115557370-3c217180-a2e4-11eb-8356-b818785630d2.png)

增加残差网络后：

![image](https://user-images.githubusercontent.com/68730894/115557436-49d6f700-a2e4-11eb-8f0b-eb9f2e83c2a1.png)

这也就是Google新论文RealFormer的核心，论文地址：https://arxiv.org/abs/2012.11747v1

![image](https://user-images.githubusercontent.com/68730894/115557541-6115e480-a2e4-11eb-9db6-ef9580e2bd26.png)


运行方案顺序
* preprocess/preprocess.py
* utils.py
* train_fine_tune.py
* predict.py
