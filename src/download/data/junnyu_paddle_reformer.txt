# paddle_reformer
Reformer: The Efficient Transformer 论文复现 paddle2.x

# requirements
- transformers
- paddlenlp
- easydict
- torch
- paddle


# 目标
ReformerModel，ReformerForSequenceClassification和ReformerForQuestionAnswering网络前向推理输出对齐参考代码。
`注`: 由于`ReformerForSequenceClassification`和`ReformerForQuestionAnswering`都使用ReformerModel作为主干部分，因此只需要对齐ReformerModel部分前向传播的权重即可。

# （1）准备
- 从 https://huggingface.co/google/ 下载reformer权重pytorch_model.bin放入google下对应的文件夹
- 从 https://huggingface.co/junnyu/reformer_paddle 下载转化后的paddle权重放入paddle下对应的文件夹

# （2）eval模式对齐
注！请进入`分析误差大的原因`文件夹的**README.md**，查看误差大的原因！！！
使用预训练的权重加载最大误差会很大，但是随即初始化模型权重，误差根本不大，这说明了误差竟然与模型初始化权重有关？？？？




# （3）train模式对齐（详细内容参考train文件夹下的`readme`，主要是额外进行了train模式下，前向传播结果对齐和反向传播的梯度对齐）
```python
# 进入train文件夹
cd train
# 在GPU和CPU模式下，进行loss和grad的比较
python compare_train.py
# test on cpu!
# compare loss
# mean difference: tensor(8.5831e-06)
# max difference: tensor(8.5831e-06)
# ==================================================
# compare grad
# mean difference: tensor(1.6347e-11)
# max difference: tensor(1.4472e-08)
# ==================================================
# test on gpu!
# compare loss
# mean difference: tensor(4.7684e-07)
# max difference: tensor(4.7684e-07)
# ==================================================
# compare grad
# mean difference: tensor(8.7412e-11)
# max difference: tensor(4.4449e-08)
```

# （4）tokenizer对齐
```python
python compare_tokenizer.py 
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
['▁I', 't', '▁is', '▁a', '▁n', 'i', 'ce', '▁d', 'ay', '▁to', 'd', 'ay', '▁', ',', '▁I', '▁w', 'ant', '▁to', '▁go', '▁to', '▁the', '▁p', 'ar', 'k', '▁', '!']
==================================================
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [33, 260, 111, 4, 136, 264, 69, 30, 71, 26, 268, 71, 258, 277, 33, 8, 180, 26, 224, 26, 13, 40, 52, 282, 258, 287], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

# 结语
- 改这个模型头都大了，其中api转换是大头！个人感觉paddle2.x的一些API好不人性化，比如gather，scatter,必须要用别的方法才能与pytorch的api对齐，之后有空再详细说下我是如何“曲线救国”的。
- 模型搭建没有出错！！！！！！误差大的原因是框架导致的！！！！！！
