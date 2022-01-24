# ChatBot with memory

This is chatbot model build using Seq2Seq model (https://arxiv.org/pdf/1409.3215.pdf) and memory block described in https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_ContextualSLU.pdf. Model is trained on Cornell Movie-Dialogs Corpus, so that 2 first lines of dialog are used to train basic Seq2Seq model, then if dialog have more lines it is divided into memory, input and output parts. E.g. if dialog has 4 lines (1, 2, 3 and 4), following data is used in training:

Memory: 1
Input: 2
Output: 3

Memory: 2
Input: 3
Output: 4

Memory: 1,2
Input: 3
Output: 4

Representation vectors produced by Seq2Seq encoder and memory block are summed and fed into Seq2Seq decoder.

# Prerequisites

Python 3

Tensorflow (tested with version 1.8.0)

# Pretrained version and running the bot

Pretrained model is available from https://drive.google.com/open?id=1axMZ1UlkW80N1ZeSAOqu0R1C3yVnTxg1. Unzip and save it in folder saved_model, then run chatbot.py. It accepts following command-line option:

* --tf cpu - default, if run with tensorflow
* --tf gpu - if run with tensorflow-gpu 
 

## Training bot

Download and unzip Cornell Movie-Dialogs Corpus (https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) into data folder, then run train.py. In the end of training model will be saved into saved_model folder. Following command-line options can be used in training:

* --tf cpu/gpu - same as in running the bot, cpu is default
* --maxvocab - maximum size of vocabulary, vocabulary is formed from training data, default 50000
* --hidden - number of units in cells used in encoder/decoder, default = 256
* --epoch - number of epochs, default =10
* --batch - size of batch, default = 128 
* --maxlen - maximum length of sentence in training batch, default = 16
* --rate - learning rate, default =0.001
* --dropout - dropout, default = 0.5
* --decrate - training is done with step decay of learning rate, factor of decay, default = 0.5
* --decstep - epoch step for reducing learning rate, default = 5, i.e. learning rate will be reduced every 5th epoch
