# Neural-Machine-Translation
English translation to French using Tensorflow seq2seq model

Use Tensorflow to build a seq2seq model and train on a small dataset of English-French sentence pairs. 
Use the whole dataset for training and inference due to data limitaion.

NMT1: Build encoder-decoder model with attention mechanism for machine translation. Use the default graph and reuse the weights between training and inference.

NMT2: Build 2 different graphs and sessions for training and inference. Share the weights by saver.

NMT3: Add visualization of training loss and two graphs by tensorboard. Use multilayer LSTM and beam search and dropout

Reference: 

https://tensorflow.google.cn/tutorials/seq2seq  

https://arxiv.org/abs/1609.08144
