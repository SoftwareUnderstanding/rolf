# Neural Language Models


This repository contains neural language model implementations trained and tested on Penn Treebank.

1. **Multi-layer LSTM with Dropout**: The link to the notebook is [here](https://github.com/pranav-ust/nlm/blob/master/notebooks/LSTM%20Language%20Model.ipynb). It receives perplexity around 80.6 on test set on default parameters.
2. **Gated Convolutional Networks with Residual Connections**: The link to the notebook is [here](https://github.com/pranav-ust/nlm/blob/master/notebooks/Gated%20Convolutional%20Networks.ipynb). It receives perplexity around 70.9 on test set on default parameters.

GCNN trains a lot faster than LSTM, due to stacked convolutions performaing parallely. However, this implementation is currently done for fixed word lengths. I am still unclear how to approach for variable lengths.

## Requirements

You will need Pytorch 0.4 and Python 3.5 to run this.


## How to run

1. For LSTM code simply run like `python3 rnn.py`
2. For GCNN code simply run like `python3 gcnn.py`

## References

### LSTM:

1. [Pytorch Language Model](https://github.com/deeplearningathome/pytorch-language-model)
2. [Offical Pytorch Tutorial on LSTM](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate/language_model)

### GCNN:

1. [Language Modeling with Gated Convolutional Networks on arXiv](https://arxiv.org/abs/1612.08083)
2. [Unofficial implementation 1 of GCNN](https://github.com/anantzoid/Language-Modeling-GatedCNN)
3. [Unofficial implementation 2 of GCNN](https://github.com/jojonki/Gated-Convolutional-Networks)


