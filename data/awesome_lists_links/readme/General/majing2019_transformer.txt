# transformer
基于tensorflow实现的论文[Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 环境配置

### docker cpu镜像

- docker pull majing/tensorflow-1.10.0-cpu-transformer:v2
- docker run -i -t -p 7000:7000 --rm majing/tensorflow-1.10.0-cpu-transformer:v2

### docker gpu镜像

- docker pull majing/tensorflow-1.10.0-gpu-transformer:v2
- docker run -i -t -p 7000:7000 --rm majing/tensorflow-1.10.0-gpu-transformer:v2

### conda环境配置

- python3.6
- pip install -r requirement.txt

## 训练en-zh的模型

### jupyter notebook交互式学习

- 在docker环境下：jupyter-notebook --no-browser --ip 0.0.0.0 --port=7000 --allow-root
- 在浏览器中访问ip:7000打开jupyter

### 预处理、训练、测试命令

- 预处理：python prepro.py
  - github中TED_data为已处理好的数据
- 训练：python train.py
- 测试：python test.py

## 论文概述

### 模型结构

<img src="image/transformer.png" width="60%" height="60%">

### 计算过程

#### 编码阶段

编码由6个相同的层组成，每一层有两个子层。第一个子层使用了多头的注意力机制，第二个子层是一个基于位置的全连接网络。在两个子层之间使用了残差连接和基于层的参数归一化。也就是说，每一个子层可以用<img src="http://latex.codecogs.com/gif.latex?LayerNorm(x + Sublayer(x))"/>来表示。

#### 解码阶段

解码也由6个相同的层组成，每一层包含三个子层，除了编码阶段两个子层外，增加了一个对编码阶段输出的多头注意力的层。在解码阶段也使用了自注意力，为了避免位置信息对解码的影响，增加了一个mask，来保证对位置i的预测只和位置i之前的输出有关系。

#### 注意力机制

注意力可以看成是query在(key, value)对的映射下，产生新的output。output是由value的加权和算出来的，权重则是根据query和key的相关度算出来的。

Scaled Dot-Product Attention是计算attention的基本单元，如下图：

<img src="image/attention.png" width="20%" height="20%">

可以用如下公式表示：
<img src="http://latex.codecogs.com/gif.latex?Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt{d_k}})V"/>

其中<img src="http://latex.codecogs.com/gif.latex?\sqrt{d_k}"/>的主要作用是为了防止<img src="http://latex.codecogs.com/gif.latex?d_k"/>比较大时，softmax的梯度会非常小，因此进行了一个归一化。这里Q、K的维度都是<img src="http://latex.codecogs.com/gif.latex?d_k"/>，V的维度时<img src="http://latex.codecogs.com/gif.latex?d_v"/>

Multi-Head Attention将注意力计算映射到不同的QKV向量，并行地产生多个<img src="http://latex.codecogs.com/gif.latex?d_v"/>维度的outputs，最终将输出concat到一起，如下图：

<img src="image/multihead.png" width="30%" height="30%">

可以用如下公式表示：<img src="http://latex.codecogs.com/gif.latex?MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O"/>，且<img src="http://latex.codecogs.com/gif.latex?head_i = Attention(QW^Q
_i, KW^K_i, VW^V_i)"/>，其中<img src="http://latex.codecogs.com/gif.latex?W^Q_i\in \mathbb{R}^{d_{model}\times d_k}"/>，<img src="http://latex.codecogs.com/gif.latex?W^K_i\in \mathbb{R}^{d_{model}\times d_k}"/>，<img src="http://latex.codecogs.com/gif.latex?W^V_i\in \mathbb{R}^{d_{model}\times d_v}"/>，<img src="http://latex.codecogs.com/gif.latex?W^O\in \mathbb{R}^{hd_{v}\times d_{model}}"/>。

在论文实验中，使用<img src="http://latex.codecogs.com/gif.latex?h=8"/>层并行计算attention，对于每一层使用参数<img src="http://latex.codecogs.com/gif.latex?d_k = d_v = d_{model}/h = 64"/>，由于每一个head的参数的维度降低了，整体计算量和之前保持一致。

模型中共有3个地方使用了这种多头注意力机制：

- encoder-decoder attention层：Q是上一层解码器的输出，K、V是encoder输出
- encoder层：自注意力，Q、K、V是上一层encoder的输出，每一层encoder的每一个位置都可以和前面层的所有位置的向量产生交互。
- decoder层：在self-attentino基础上使用了masking。

#### 前向网络

在encoder和decoder输出上进行如下计算：
<img src="http://latex.codecogs.com/gif.latex?FFN(x) = max(0, xW_1 + b_1)W_2 + b_2"/>

对于同一层的不同位置，参数是共享的；不同层间参数不共享。

也可以把这个理解成两个<img src="http://latex.codecogs.com/gif.latex?kernel\_size=1"/>的卷积，输入和输出的维度都是<img src="http://latex.codecogs.com/gif.latex?d_{model}=512"/>，中间层维度是<img src="http://latex.codecogs.com/gif.latex?d_{ff}=2048"/>

#### Embedding和Softmax

输入token、输出token到向量的转换过程使用共享的embedding。

#### 位置编码

由于整个模型没有使用循环或者卷积操作，在encoder和decoder中加入位置编码。位置编码计算方法如下：

<img src="http://latex.codecogs.com/gif.latex?PE(pos,2i) = sin(pos/10000^{2i/d_{model}})"/>

<img src="http://latex.codecogs.com/gif.latex?PE(pos,2i+1) = cos(pos/10000^{2i/d_{model}})"/>

其中pos是位置，i是维度（即向量中第i个元素）。那么对任意位置k，
<img src="http://latex.codecogs.com/gif.latex?PE_{pos+k}"/>可以表达成<img src="http://latex.codecogs.com/gif.latex?PE_{pos}"/>的线性函数。