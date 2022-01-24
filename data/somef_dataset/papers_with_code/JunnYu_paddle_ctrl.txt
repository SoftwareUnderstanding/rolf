# CTRL: A Conditional Transformer Language Model for Controllable Generation
paddle2.x 实现 [CTRL](https://arxiv.org/pdf/1909.05858.pdf)

# 准备工作
- 下载预训练权重（pytorch版本的，https://huggingface.co/ctrl/ 和 https://huggingface.co/sshleifer/tiny-ctrl）,放入hg/文件夹对应目录。
- 【可选/推荐】手动转换：`python convert.py`
- 【不推荐】百度云下载：链接：https://pan.baidu.com/s/1dnRwCRClqsXvG8475v2m9g 提取码：ogn8 

# CTRLLMHeadModel前向对齐

```bash
python compare_lm.py
############ sshleifer-tiny-ctrl
# compare loss:
# mean difference: tensor(0.)
# max difference: tensor(0.)
# compare logits:
# mean difference: tensor(1.2408e-08)
# max difference: tensor(5.6252e-07)
# compare hidden_states:
# mean difference: tensor(4.2710e-08)
# max difference: tensor(3.2783e-06)
# mean difference: tensor(4.3044e-08)
# max difference: tensor(3.2783e-06)
# mean difference: tensor(1.2185e-07)
# max difference: tensor(4.5300e-06)

############ 加载ctrl权重误差非常大,进入文件夹【探究ctrl原版权重误差大的原因】查看原因
```

# CTRLForSequenceClassification前向对齐

```bash
python compare_cls.py
############ sshleifer-tiny-ctrl
# compare loss
# mean difference: tensor(2.3842e-07)
# max difference: tensor(2.3842e-07)
# compare logits
# mean difference: tensor(6.0303e-09)
# max difference: tensor(1.4901e-08)

############ 加载ctrl权重误差非常大,进入文件夹【探究ctrl原版权重误差大的原因】查看原因
```


# tokenizer对齐
```bash
python compare_tokenizer.py
# input_ids:      True
# token_type_ids: True
# attention_mask: True
```

# 比较生成的文本

## paddle版本生成结果：`PADDLE-GENERATE.ipynb`
```python
Diet English : I lost 10 kgs! ; German : 
 Ich habe zehn Kilogramm abgenommen! 
 
 Als ich das erste Mal mit meinem Smartphone war, war es ein wenig schwierig zu finden, wo man die App herunterladen kann. Aber jetzt ist sie da. 
 
 Das Smartphone hat mich auch sehr beeindruckt. Es machte mir viel Spaß. Und so funktioniert mein Leben heute ganz einfach und ohne große Probleme. 
 
 Mein Fazit: Wenn du deine Apps auf dem iPhone oder Android
==================================================
Reviews Rating: 5.0
 I have been using this product for a few years now and it is the best thing on the market to keep your teeth white. It does not taste bad at all like some of these other products do. The only problem with this product is that you need to use it every day or else they will start coming back in after about 2 weeks. But if you do that, then it's worth it. You can also buy them from Amazon but shipping takes forever. So just make sure you order enough so you don't run out. 
 Rating: 5.0 
 This stuff works great. My dentist recommended it, and I'm glad he did. It's easy to use, tastes good, and
==================================================
Questions Q: What is the capital of India?
 A: mumbai. 
 Q: Who was a British politician who served as Prime Minister from 1922 to 1924? 
 A: edward viibert 
 Q: The name of which city in New South Wales has been used for many years by the Australian National Football team? 
 A: sydney 
 Q: Which American actor starred with his wife and daughter on the television series 'Family Affair'? 
 A: james coburn 
 Q: In what year did the first edition of this book appear? 
 A: 1962 
 Q: How long does it take to make one pound of sausage? 
 A: 24 hours
==================================================
Books Weary with toil, I haste me to my bed,
 And sleep till the morning of life is come. 
 The sun has risen and his beams are bright, 
 But still he shines upon a world forlorn; 
 He sees no more its joys or griefs below, 
 Nor hears their murmur as they pass below. 
 My heart grows weary for the world's delight, 
 For all that makes it dear in human eyes; 
 It feels like one who wanders through an empty land, 
 With nothing left but desolation there. 
 O God! how long shall this be mine abode, 
 Where every joy hath passed away from me? 
 How long, O God, must I thus wander here, 
 In sorrow
==================================================
```

## huggingface版本生成结果：`PYTORCH-GENERATE.ipynb`
```python
Diet English : I lost 10 kgs! ; German : Ich habe zehn Kilogramm abgenommen! 
 
 Als ich das erste Mal mit meinem Smartphone war, war es ein wenig schwierig zu finden, wo man die App herunterladen kann. Aber jetzt ist sie da. 
 
 Das Smartphone hat mich auch sehr beeindruckt. Es machte mir viel Spaß. Und so funktioniert mein Leben heute ganz einfach und ohne große Probleme. 
 
 Mein Fazit: Wenn du deine Apps auf dem iPhone oder Android
==================================================
Reviews Rating: 5.0 
 I have been using this product for a few years now and it is the best thing on the market to keep your teeth white. It does not taste bad at all like some of these other products do. The only problem with this product is that you need to use it every day or else they will start coming back in after about 2 weeks. But if you do that, then it's worth it. You can also buy them from Amazon but shipping takes forever. So just make sure you order enough so you don't run out. 
 Rating: 5.0 
 This stuff works great. My dentist recommended it, and I'm glad he did. It's easy to use, tastes good, and
==================================================
Questions Q: What is the capital of India? 
 A: mumbai. 
 Q: Who was a British politician who served as Prime Minister from 1922 to 1924? 
 A: edward viibert 
 Q: The name of which city in New South Wales has been used for many years by the Australian National Football team? 
 A: sydney 
 Q: Which American actor starred with his wife and daughter on the television series 'Family Affair'? 
 A: james coburn 
 Q: In what year did the first edition of this book appear? 
 A: 1962 
 Q: How long does it take to make one pound of sausage? 
 A: 24 hours
==================================================
Books Weary with toil, I haste me to my bed, 
 And sleep till the morning of life is come. 
 The sun has risen and his beams are bright, 
 But still he shines upon a world forlorn; 
 He sees no more its joys or griefs below, 
 Nor hears their murmur as they pass below. 
 My heart grows weary for the world's delight, 
 For all that makes it dear in human eyes; 
 It feels like one who wanders through an empty land, 
 With nothing left but desolation there. 
 O God! how long shall this be mine abode, 
 Where every joy hath passed away from me? 
 How long, O God, must I thus wander here, 
 In sorrow
==================================================
```

# 注意：
- paddlenlp中未实现`RepetitionPenaltyLogitsProcessor`,本项目使用两种方法实现该方法。
- paddlenlp的generate与huggingface的generate有些许区别，huggingface的generate时候max_length包含你输入的文本，而paddlenlp的generate的max_seq_len不包括，因此我设置最大长度时候减去了文本长度。

```python
def __call__(self, input_ids, logits):
    # method 1 使用了for循环，好理解，速度不一定快
    # score = paddle.index_sample(logits, input_ids)
    # score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
    # outputs = [paddle.scatter(logit,input_id,score_) for logit,input_id,score_ in zip(logits,input_ids,score)]
    # return paddle.stack(outputs,axis=0)

    # method2 与之前reformer类似，同样采用了添加offset的方法，使用scatter对flatten后的输入进行操作。
    score = paddle.index_sample(logits, input_ids)
    score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
    input_ids = input_ids + paddle.arange(logits.shape[
        0]).unsqueeze(-1) * logits.shape[-1]
    return paddle.scatter(logits.flatten(),input_ids.flatten(),score.flatten()).reshape(logits.shape)
```
# Reference

```bibtex
@article{keskar2019ctrl,
  title={Ctrl: A conditional transformer language model for controllable generation},
  author={Keskar, Nitish Shirish and McCann, Bryan and Varshney, Lav R and Xiong, Caiming and Socher, Richard},
  journal={arXiv preprint arXiv:1909.05858},
  year={2019}
}
```

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```