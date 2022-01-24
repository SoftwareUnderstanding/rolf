# Attention

## 博客知识

1.RNN,Seq2Seq,Attention简介：<br>
[完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)<br>
[补充知识：LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)<br>

2.Attention存在的问题：<br>
(I)对噪声敏感<br>
(II)对长句识别性能下降<br>
[GMIS 2017 | 腾讯AI Lab副主任俞栋：语音识别研究的四大前沿方向](https://mp.weixin.qq.com/s?__biz=MzIzOTg4MjEwNw==&mid=2247483689&idx=1&sn=48c06c6cf270dc6b9db5ae46f78e520c&scene=21#wechat_redirect)<br>

3.解读transformer：<br>
[Attention is All You Need | 每周一起读](https://zhuanlan.zhihu.com/p/27600655)<br>

4.解读CNN机器翻译：<br>
[Facebook提出全新CNN机器翻译：准确度超越谷歌而且还快九倍](https://zhuanlan.zhihu.com/p/26817030)<br>

## 论文
1.attention用于语音识别前的baseline（用于翻译），对于长句识别效果急剧变差：<br>
[Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning
to align and translate. arXiv:1409.0473, September 2014](https://arxiv.org/pdf/1409.0473.pdf)

2.第一篇将attention用于语音识别的文章(提出hybird attention机制，解决attention的位置信息问题，提出`attention加窗机制`，数据集：`TIMIT`)：<br>
[J.Chorowski, D.Bahdanau, D.Serdyuk, K.Cho, and Y.Bengio."Attention-based models for speech recognition"](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)<br>

3.attention加窗改进，RNN更换为GRU在LVCSR任务上的应用（数据集：`(WSJ) corpus (available as LDC93S6B and LDC94S13B)`）：<br>
[D. Bahdanau, J. Chorowski, D. Serdyuk, P. Brakel, and Y. Bengio,“End-to-end attention- based large vocabulary speech recognition,”](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7472618)<br>

4.最经典的attention语音识别模型：<br>
[W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: Aneural network for large vocabulary conversational speech recognition”](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7472621)<br>

5.CTC+Attention(CTC强化了对齐的单调性，且CTC加快网络训练速度，数据集：`WSJ1 (81 hours)` ,`WSJ0 (15 hours) `, `CHiME-4 (18 hours)`):<br>
[S. Kim, T. Hori, and S. Watanabe, “Joint CTC-attention based end-to-end speech recognition using multi-task learning”](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7953075)<br>

6.transformer:<br>
[N. Shazeer, Niki Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin "Attention Is All You Need"](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)<br>

7.CNN机器翻译：<br>
[J. Gehring, M. Auli, D. Grangier, D. Carats, Y. N. Dauphin "Convolutional Sequence to Sequence Learning"](http://delivery.acm.org/10.1145/3310000/3305510/p1243-gehring.pdf?ip=61.150.43.51&id=3305510&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E1DE562CDF7C9BB11%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1576586699_b9d77762bc10a1c4d4c4da49c7d10881)<br> 

8.CNN+RNN的E2E ASR：<br>
[VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION](https://arxiv.org/pdf/1610.03022.pdf)

9.CNN+RNN attention+CTC（Encoder在RNN前加入VGGnet进行特征提取，数据集：`WSJ0`,`CHIME-4`）:<br>
[Kim S, Hori T, Watanabe S. Joint CTC-attention based end-to-end speech recognition using multi-task learning](https://arxiv.org/pdf/1609.06773.pdf)<br>

10.self-attention+CTC:<br>
[SELF-ATTENTION NETWORKS FOR CONNECTIONIST TEMPORAL CLASSIFICATION IN SPEECH RECOGNITION](https://arxiv.org/pdf/1901.10055.pdf)

## 代码
1.[tensorflow E2EASR代码](https://github.com/hirofumi0810/tensorflow_end2end_speech_recognition)<br>

2.[pytorch E2EASR代码](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)<br>
需要解决的问题：如何多GPU运行
