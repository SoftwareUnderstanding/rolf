# Sarcasm Detection

This project implements sarcasm detecrion by CNN(Shuhui Wu) and Bert(Shuqi Gao)

## Dataset
* We obtain dataset of Labelled Sarcastic Comments from Kaggle: https://www.kaggle.com/sherinclaudia/sarcastic-comments-on-reddit
* In CNN implementation, we use pretrained GloVe word embedding: https://nlp.stanford.edu/projects/glove/
* In BERT, we use google's pretrained bert by two different package:https://github.com/google-research/bert, https://github.com/huggingface/transformers

## Approach
* CNN: https://www.aclweb.org/anthology/N18-2018.pdf &nbsp; &nbsp; &nbsp; notebook: &nbsp;Final_proj_CNN.ipynb
* BERT: papper: https://arxiv.org/pdf/1810.04805v2.pdf 
  <br>followed:&nbsp;&nbsp;https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb 
<br>notebook:&nbsp;try_bert_1.ipynb and try-bert-2.ipynb
  
