# Comparison of Unsupervised Data Augmentation with BERT and XLNet
## Final Project for W266: Natural Language Processing with Deep Learning
### Thomas Goter
### Fall 2019

The Unsupervised Data Augmentation (UDA) methodology presented in https://arxiv.org/pdf/1904.12848.pdf is further extended beyond its initial use with BERT for use with the [XLNet](https://arxiv.org/pdf/1906.08237.pdf) transformer-based neural language model for evaluation on a novel text classification dataset. The results discussed herein show that the benefits of UDA are reproducible and extensible to other modeling architectures, namely XLNet. For the novel *Song of Ice and Fire* text classification problem presented herein, absolute error rate reductions of up to 5\% were shown to be possible with an optimized UDA model.  Additionally, it is shown UDA can achieve the same accuracy as a finetuned model with as little as 67\% of the labeled data. However, UDA is not a magic bullet for enabling the use of complex neural architectures for cases with very limited sets of labeled data, and the complexity of its use and time associated with its optimization should be weighed against the cost of simply labeling additional data.

![alt text](https://github.com/tomgoter/nlp_finalproject/blob/master/report/working/uda_errors.png)

**Process**
 - Pretrained BERT and XLNet models are not stored in this repo given their size. These were downloaded as given below.
 - BERT: wget -P /content/drive/My\ Drive/NLP_FinalProject/working/bert_pretrained https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
 - XLNET: Link for local download provided on [XLNet GitHub repository](https://github.com/zihangdai/xlnet) (used Cased, Base model)
 
 - Processed Game of Thrones data for the various data set sizes are available in the following directories: 
 - BERT: ./Data/proc_data/GoT 
 - XLNET: ./Data/proc_data/GoT_xlnet
 
 - Pretrained models and Data is warehoused in personal Google Drive and Google Bucket for running with GPUs and TPUs through Google's colaboratory environment. 
 
 - The final run script (bert_xlnet_uda_colab.ipynb) is set up to run on either GPUs or TPUs and can run either XLNet or BERT models. There are default options provided for each. And all options can be overwritten in the notebook. 
 
 - Typical model runtimes range between 30-60 minutes on an 8-core TPU. 

**Acknowledgement**  
This work builds on the work presented in https://arxiv.org/pdf/1904.12848.pdf and extends it to optionally make use of the [XLNet](https://arxiv.org/pdf/1906.08237.pdf) architecture as well. As such, much of the open source code from the respective GitHub repositories for [UDA](https://github.com/google-research/uda) and [XLNet](https://github.com/zihangdai/xlnet) was leveraged for this project. Thank you!


