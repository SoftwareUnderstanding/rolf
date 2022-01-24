# ChatRobot

## 0. 特别改进提醒  
> * 关于keras环境下seq2seq错误修改  
  (from keras.engine.base_layer import Node,_collect_previous_mask, _collect_input_shape)  
> * 0.1 将blilstm_cnn_crf.py代码中merge改为Concatenate，保证网络拼接正确  
> * 0.2 引入了earllystopping 可能使得model提前终止，注意调参
> * 0.3 引入了trig位置embdding，在encode阶段引入，收敛速度加快

## 1. 效果展示  
### 1.0 `python train.py`执行效果图  
> * 注意：aucc可能很小，因为embdding的维度大，完全一样不可能，只能比较mse
![image](./images/train.jpg)  
### 1.1 `python test.py`执行效果图  
![image](./images/test-ans.jpg)  

## 2. 执行命令  
> * 生成序列文件,将文字编码为数字,不足补零，执行训练样本之前执行
`python data_process.py`  
> * 生成word2vec向量,包括编码向量和解码向量  
`python word2vec.py`  
> * 训练网络  
`python train.py`  
> * 测试  
`python test.py`  
> * 模型评分  
`python score.py`  
> * 智能问答  
`python chat_robot.py`  
> * 绘制word2vec向量分布图  
`python word2vec_plot.py`  
> * 直接jupyter运行预测结果
`predict_test.ipynb`

## 3. 更新 或者用jieba分词
> * Word2cut模型对陌生词汇的分词未解决,后续会补齐jieba分词.

## 注意感谢，以上很多内容需感谢@shen1994,有不妥之处请联系：976344083@qq.com
> * seq2seq论文地址: https://arxiv.org/abs/1409.3215  
> * seq2seq+attention论文地址: https://arxiv.org/abs/1409.0473  
> * ChatRobot启发论文: https://arxiv.org/abs/1503.02364
> * seq2seq源码: https://github.com/farizrahman4u/seq2seq  
> * seq2seq源码需求: https://github.com/farizrahman4u/recurrentshop  
> * beamsearch源码参考: https://github.com/yanwii/seq2seq
> * bucket源码参考: https://github.com/1228337123/tensorflow-seq2seq-chatbot-zh

