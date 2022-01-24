NLP_and_DL_practic
===
This repo contains tutorials covering understanding and implementing NLP related Model.  
And it has some common requirements in below for implementing this Model.  

## **Requirements** 
```bash
-python-3.7.9  
-pytorch==1.7.0+cu101  
-torchtext==0.3.1  
-spacy==2.2.4  
```  
  
    
## **Practices**
*1.[ConvNets and CNN-LSTM Sentiment Analysis Practice](https://github.com/yinghao1019/NLP_and_DL_practice/blob/master/Convolution_Neural_Netowrks_for_sentence_classification_Practice.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/yinghao1019/NLP_and_DL_practice/blob/master/Convolution_Neural_Netowrks_for_sentence_classification_Practice.ipynb)*  
  
*This pratice implementation is referenced to [Convolutional Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb).Besides original notebook's Convolution Model,we create extened Model using RNN_layer and traditional ML model,and on top of that we testing different type hyperparmeter tuning for create Model.  

2.[NMT using seq2seq(Lstm) practice](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/NMT_jointLearn(Prac).ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/NMT_jointLearn(Prac).ipynb)*  
  
*This pratice implementation is referenced to [ Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb).  
  
3.[NMT using Pack pad sequence practice](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/Packed_PAD(Prac).ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/Packed_PAD(Prac).ipynb)*  
  
*This pratice implementation is referenced to [ Packed Padded Sequences, Masking, Inference and BLEU](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/Packed_PAD(Prac).ipynb). 

4.[Conv_seq2seq](https://github.com/yinghao1019/NLP_and_DL_practice/blob/master/Conv_seq2seq.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinghao1019/NLP_and_DL_practice/blob/master/Conv_seq2seq.ipynb)*  
  
*This practice is references to [Convolutional Sequence to Sequence Learning](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb).Difference from original notebook,We used exponetial learning rate decay to training lanuage Model stable.  


  
## **Reference**  
Below describtion will display Referenced paper by each pratice.if you are very interseting,welcome to trace these papers.  

1.[Convolutional Neural Networks for Sentence Classification ](https://arxiv.org/abs/1408.5882)  
2.[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
4.[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)

