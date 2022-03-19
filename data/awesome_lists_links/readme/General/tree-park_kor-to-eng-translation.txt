# Korean to English Translation
Translation task implementation with transformer and seq2seq attention mechanism with low level pytorch code


## Tokennizer  
- Korean : pip install konlpy
- English : pip install nltk


## Architecture  
BiLSTM Seq2Seq with Attention
- Word Embedding
- BiLSTM Encoder or Stacked LSTM Encoder
- LSTM Decoder with Attention mechanism

Transformer
- Positional Encoding + WordEmbedding
- Multi-head Attention, Position-wise Fead Forward Network
- ResNet + NormLayer


## Refer
- Attention mechanism : Neural Machine Translation by Jointly Learning to Align and Translate (https://arxiv.org/abs/1409.0473)
- Transfomer: Attention is all you need (https://arxiv.org/abs/1706.03762) 


## Start
pip3 install -r requirements.txt
