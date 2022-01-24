## 0.Requirements
* Ubuntu14.04 LTS
* Python 3.5+
* PyTorch 0.3+
* pandas
* tqdm
* cnn_finetune(https://github.com/creafz/pytorch-cnn-finetune)

### 1.使用基于ImageNet训练的nasnetalarge模型
* 模型链接：http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth
* 模型论文：Zoph B, Vasudevan V, Shlens J, et al. Learning Transferable Architectures for Scalable Image Recognition[J]. 2017.(https://arxiv.org/abs/1707.07012)
* 开源模型在算法中作用：提供模型优化的初始参数及网络结构设置。

## 2.Folder：
	|--data
		|--guangdong_round1_train1_20180903
		|--guangdong_round1_train2_20180916
		|--guangdong_round1_test_a_20180916
		|--guangdong_round1_test_b_20181009
	|--code
		|--gen_label_csv.py
		|--train-cnn-main.py

* 注:由于Ubuntu14.04默认没有中文支持，图片数据需在Windows下解压后再scp到Ubuntu方可访问。

## 3.run：
### Option1:using pretrain model to predict
	python train-cnn-main.py

### Option2:training model to predict
	python train-cnn-main.py --test False

