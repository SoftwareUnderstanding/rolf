## Finetuning_BART_for_FACET_Summarization (In Process...)

## BART FACET PAPER [https://aclanthology.org/2021.acl-short.137/](https://aclanthology.org/2021.acl-short.137/)

## 1) Introduction

BART model [https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)

Fairseq [https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq)

Fairseq tutorial on fine-tuning BART on Seq2Seq task [https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)

Emerald dataset used for faceted summarization - https://github.com/hfthair/emerald_crawler


## 3) Fine-tuning

Prerequisite:

**PyTorch**

**Fairseq** 

**Download the pretrained BART large model**

** Get the emerald dataset **

#Preprocessing data

  python preprocess_data.py
	bash bpe.sh
	bash binarize.sh

# for parameter check the finetune.sh script 

#Although we did not find major differene with updating the max_tokens in the bart finetuning in case you want to try the code changes for allowing more than default tokens is in train.py file

 scripts/train.py
 update from line 157 

Find more information from [fairseq bart repo](https://github.com/pytorch/fairseq/tree/master/examples/bart)!
