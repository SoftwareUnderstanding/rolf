# Machine_Translation_Seq2Seq
Machine translation (jp-en) using LSTM-based encoder-decoder model (Pytorch). This is the implementation of several models:
- model.py https://arxiv.org/abs/1409.3215, 
- model2.py https://arxiv.org/abs/1406.1078

adapted to JP-EN translation.
  
Data: https://nlp.stanford.edu/projects/jesc/, official split. The xls data is converted into csv with panda (prepro.py). Japanese is tokenized using sentencepiece (https://github.com/google/sentencepiece/), English is tokenized using space (sorry, too lazy).
