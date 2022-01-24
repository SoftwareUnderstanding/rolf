# NLP
NLP 最新进展的论文实现

本人提供技术支持：1007171627@qq.com

## 一、Transformer 
数据集参考http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/
实现了《Attention is all your need》 中提到的算法。目前的进度，完成训练代码，测试代码但是并没有进行测试。
论文网址：https://arxiv.org/pdf/1706.03762.pdf
### 1.效果
    iwslt2016 数据验证BLEU = 26.58, 57.5/34.1/21.6/13.9 (BP=0.960, ration=0.960)
### 2.学习过程
![](https://github.com/caoyujiALgLM/NLP/blob/master/loss.jpg)
    
### 3.数据准备
#### 3.1下载下面的数据到./dat/nmt/iwslt2016
    链接: https://pan.baidu.com/s/1HETZpdDyArvlh4OjpefvWQ 提取码: ust5 复制这段内容后打开百度网盘手机App，操作更方便哦
#### 3.2执行
    python3.6 dat/nmt/sentence_piece_model.py
### 4.训练脚本
```shell
#!/usr/bin/env bash
pwd

export PYTHONPATH=$PYTHONPATH:./src
export CUDA_VISIBLE_DEVICES=0

nohup python3.6 src/nmt/transformer/train.py \
    --batch_size=128 \
    --num_train_samples=196884 > log_train.txt 2>&1 &
```
### 5.测试脚本
    #!/usr/bin/env bash
    export PYTHONPATH=$PYTHONPATH:./src
    export CUDA_VISIBLE_DEVICES=1

    # 执行eval
    python3.6 ./src/nmt/transformer/test.py \
        ./out/nmt/transformer/y_predict.txt \
        ./out/nmt/transformer/y_label.txt \
        1 \
        9990

    # 计算blue 值
    perl ./src/nmt/multi-blue.perl \
        ./out/nmt/transformer/y_label.txt < ./out/nmt/transformer/y_predict.txt
### 6. 当前训练参数不适合大型训练集：训练参数参考https://arxiv.org/pdf/1804.00247.pdf
### 7.下一个版本即将引入的新功能
- 基于tensorflow 运算实现greedy search
- 基于tensorflow 运算实现beam search

## 二、NER for Chinese
实现了基于RNN/CNN/IDCNN + CRF的中文命名实体识别。

### 结果
#### Loss-Function
        	SoftMax CRF
	BiLSTM	90.2pp	92.2pp
#### 模型横评
		BiLSTM-CRF	CNN-CRF	IDCNN-CRF
	f1	92.2pp	90.4pp	90.2pp
#### 标注方法
	        	BMES	BIOS
	BiLSTM-CRF	91.7pp	92.2pp

## 三、句对分类模型
实现了基于Co-attention BERT 的句对分类模型，目前模型准确度93.1pp。


