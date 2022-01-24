### 论文阅读笔记样例

**REFORMER：THE EFFICIENT TRANSFORMER** [arxiv](https://arxiv.org/abs/2001.04451)

**论文小结：**作者提出了一种新的转换器模型，对体系架构进行了两项重大改进：

* 1）使用可逆层以防止需要存储所有层的激活来进行反向传播；
* 2）使用局部敏感哈希来估算耗时间的softmax计算。该Reformer的性能与SOTA Transformer模型相当，但内存效率更高，长序列的速度更快。

**代码实现**: [github](https://github.com/google/trax/tree/master/trax/models/reformer)