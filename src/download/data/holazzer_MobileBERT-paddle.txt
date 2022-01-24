# MobileBERT paddle

使用 `paddlepaddle` 实现 `transformers` 中提供的 `MobileBERT` 模型。


论文： **MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**

https://arxiv.org/abs/2004.02984


huggingface 模型页面：  

https://huggingface.co/google/mobilebert-uncased


transformers 源代码：

https://github.com/huggingface/transformers/tree/master/src/transformers/models/mobilebert


## 论文和代码解析

本文对bert模型进行了知识迁移，把大模型的知识迁移到小模型中。

深层网络会很难训练，尤其是在小模型中我们把模型的“腰”收的很紧，这样就更不容易训练了。所以这里作者采取的方法是，先训练一个大尺寸的网络作为教师，然后在小模型（学生）网络的设计中，把每一层的 feature map 设计成相同的形状。这样，就可以在训练时，让这两个模型尽量对齐。

### 模型设计

这里结合论文和代码，对模型设计进行介绍。我想用一个先总后分的方法来讲。

先自上而下，讲解模型的构造，再从下到上，看每一个子网络的实现。

```python
class MobileBertModel(MobileBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = MobileBertEmbeddings(config)
        self.encoder = MobileBertEncoder(config)
        self.pooler = MobileBertPooler(config) if add_pooling_layer else None
```

`MobileBertModel` 包括 embedding 和 encoder，以及一个 pooler。
（原文中没有 pooler， 所以我们一会儿先看看 pooler是干什么的。）

![](static/table_1.png)




### 🌊 Pooler

我们先看这个“可有可无”的 pooler:

```python
class MobileBertPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.do_activate = config.classifier_activation
        if self.do_activate:
            self.dense = nn.Linear(512, 512)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        else:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = paddle.tanh(pooled_output)
            return pooled_output
```

解释：encoder最后会输出 `(batch_size, 24, 512)` 维的向量。

24是body的数量，512是设置的embedding维度，MobileBERT的body设计刚好是进512出512。

这里pooler的意思是直接拿第一个512作为模型输出，或者是再加一层 Linear，还是输出 512。 

因此，这个 Pooler 确实是 “可有可无”， 哈哈。


### 🍩 Embedding

```python
class MobileBertEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.trigram_input = True
        self.embedding_size = 128
        self.hidden_size = 512

        self.word_embeddings = nn.Embedding(config.vocab_size, 128, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, 512)
        self.token_type_embeddings = nn.Embedding(2, 512)

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = nn.Linear(embedded_input_size, 512)

        self.LayerNorm = NoNorm(512)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

（其中部分 config 项被我换成了数字。）
很清楚，一个word embedding，一个 position embedding，一个 token type embedding。

```python
    def forward(self, input_ids, token_type_ids, position_ids, inputs_embeds):
        if self.trigram_input:
            inputs_embeds = paddle.concat([
                nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0),
                inputs_embeds,
                nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0),
            ], axis=2)

        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

（为方便阅读，部分删去。）

用3-gram的话，就要把 input 左错开1位，右错开1位，concat起来，就得到了3-gram向量。

再过一个 embedding 变成512。MobileBERT的一个设计就是输入的embedding尺寸是128，经过一个fc变成512。


### 🍭 Encoder

Encoder 是模型的主干部分。

```python
class MobileBertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.LayerList(
            [MobileBertLayer(config) for _ in range(24)])
```

论文中的图果然诚不欺我，真的就是24个body的部分串起来。

```python
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,     # 是否要输出 attention
            output_hidden_states=False,  # 是否要输出隐藏状态
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None


        for i, layer_module in enumerate(self.layer):  # 每次过一个layer
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
            )

            hidden_states = layer_outputs[0]  # <-  隐藏状态用于下一层输入

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # <- 本层 attention 

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if
                         v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=all_hidden_states,
                               attentions=all_attentions)
```

很直观，就是把24个 Layer 层过了一遍，每一层会输出自己的 hidden state，最后一层的 hidden state 作为encoder 最终输出。


### 🎄 MobileBERTLayer

```python
class MobileBertLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = True
        self.num_feedforward_networks = 4

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck: self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.LayerList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])
```

这里要参考上面论文里的图。

![](static/table_1_mobile_bert.png)

这里有几处 MobileBERT 的设计需要说一下：

（1）宽进宽出中间窄，为了方便和教师模型的输出对齐。

![](static/fig_1.png)

用大白话说，MobileBERT就是把中间变窄了。
Figure 1(b)是teacher，它走两条路，在图中标为红色和蓝色，这两条路就是1(a)中普通BERT的路，关键就在于MobileBERT的设计是图中话出来的路尺寸是一样的，但是中间的肚子尺寸不一样，这样就可以节约模型的尺寸了。

注意观察(b)和(c)的Linear开口方向不一样，teacher是从小到大，利于模型的训练；mobile是从大到小，压缩模型尺寸。

（2）Stacked FFN

MobileBERT 把肚子缩小了，这样导致 MHA 和 FFN 的参数比例发生了变化，这样信息的表示就不匹配了，所以作者把FFN多加了几层。这里是从1层变成了4层叠加。"carefully balanced".

（3）NoNorm

为了节约计算，MobileBERT 把所有的 Layer Normalization 换成了 NoNorm 操作，具体参见原文。NoNorm 操作计算的是 Hadamard 积，也就是两个矩阵按位置分别相乘。

```python
class NoNorm(nn.Layer):
    def __init__(self, feat_size: Union[int, Tuple], eps=None):
        super().__init__()

        if isinstance(feat_size, int): feat_size = (feat_size, )

        bias = paddle.zeros(feat_size)
        weight = paddle.ones(feat_size)

        self.bias = paddle.create_parameter(
            shape=bias.shape, dtype=bias.dtype,
            is_bias=True,
            default_initializer=paddle.nn.initializer.Assign(bias)
        )

        self.weight = paddle.create_parameter(
            shape=weight.shape, dtype=weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(weight)
        )

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias  # Hadamard product
```

这里的好处就比较玄妙了，反正我不是特别明白。。。显而易见的是计算比较方便，其他方面就不太清楚了。。。另外，模型用relu而不是gelu。



好了，现在看一下 Layer 内的 forward 操作：
```python
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        # 1. 过 bottleneck （FC）
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        # 2. 过 MHA
        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 3. 过 stacked FFN
        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        # 4. 恢复尺寸的 FC
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)

        outputs = (
            (layer_output,)
            + outputs
            + (
                paddle.to_tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs
```

接下来就这4个部分来看每个模块。

#### 🍼 Bottleneck

```python
class BottleneckLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NoNorm()

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input

class Bottleneck(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = True
        self.use_bottleneck_attention = False
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def forward(self, hidden_states):
        bottlenecked_hidden_states = self.input(hidden_states)

        shared_attention_input = self.attention(hidden_states)
        return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
```

其实Bottleneck就是FC，上面红蓝两条路，这里对应两个不同用途的 BottleneckLayer。


#### 🍩 Attention

```python
class MobileBertAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = MobileBertSelfAttention(config)
        self.output = MobileBertSelfOutput(config)

class MobileBertSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

class MobileBertSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        self.LayerNorm = NoNorm()
```

MHA每个是4个head，在 SelfAttention 中实现了。最后一个 FC + NoNorm 输出 SelfOutput 。


#### 🍰 FFN

```python
class MobileBertIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.relu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class FFNOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NoNorm()

    def forward(self, hidden_states, residual_tensor):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor) # Add & Norm
        return layer_outputs

class FFNLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs
```

Intermediate 就是 FC + Relu。为什么叫 Intermediate 呢？这是transformers实现里面的名字，我认为它的意思是作为两个维度的中介，就是为了转换维度使用的 FC 。

FFNOutput 实现了 Add & Norm。

没啥说的了，图里有，FFN = [128 -> 512 -> 128] * 4

最后一层的 Linear 用的就是Intermediate。

最后，从小肚子再恢复到512需要反向的bottleneck结构，基本上就是前面的bottleneck反过来，所以我偷下懒就不说了，可以自己看代码。

大功告成，MobileBERT的网络结构基本就是这样了。具体细节参考代码中的 forward 操作。


## 与 😀huggingface 上的预训练模型进行对齐

```python
# transformers pytorch 版本
from transformers.models.mobilebert import MobileBertModel, MobileBertTokenizer

mb = MobileBertModel.from_pretrained('google/mobilebert-uncased')

tk = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

sentence = "Advancing the state of the art: We work on computer science problems that define the technology of today and tomorrow."

i = tk(sentence, return_tensors='pt')

mb.eval()
o = mb(**i)
```


```python
# paddle 版本
import paddle
from mobile_bert_model import *
from config import MobileBertConfig
from ppnlp_tokenizer import MobileBertTokenizer

config = MobileBertConfig()
model = MobileBertModel(config, add_pooling_layer=False)
pretrained_weights = paddle.load('path/to/your/converted/weight')
model.load_dict(pretrained_weights)
tokenizer = MobileBertTokenizer('./mobilebert-uncased/vocab.txt', do_lower_case=True)
# 使用 ppnlp 的 tokenizer 加载了预训练模型的 vocab 文件，可以兼容

def tk(s):
    d = tokenizer(s)
    for k, v in d.items():
        d[k] = paddle.to_tensor((v, ))
    return d

sentence = "Advancing the state of the art: We work on computer science problems that define the technology of today and tomorrow."

i = tk(sentence)

model.eval()
o = model(**i)
```

p.s. "Advancing the state of the art: We work on computer science problems that define the technology of today and tomorrow." 是 google 在 huggingface 上写的 Research interests.


输出：

![](static/align_res.png)

上方是transformers的输出结果，下方是paddle版的输出结果。

由于float的精度问题，float是数值越大越不精确，因此这个输出的数值很大的时候，精度就会很差。。。

如果只看有效数字位的话，精度还是不错的，但是由于这是float，越大在数轴上越稀疏，这数肯定就越不精确了。虽然我们看着数基本上都对，实际上用 allclose 比较时，精度只有`rtol=0.75`。。。


