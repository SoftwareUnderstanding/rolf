# Transfomer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ieLHIoP1GYBbHtdAyiIwcuSwy2TtAk5A)
This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). This model is based solely on attention mechanisms and introduces Multi-Head Attention. The encoder and decoder are made of multiple layers, with each layer consisting of Multi-Head Attention and Positionwise Feedforward sublayers. This model is currently used in many state-of-the-art sequence-to-sequence and transfer learning tasks.  
<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>  

## Note on Attention in the model
The Transformer uses multi-head attention in three different ways:   
1) In “encoder-decoder attention” layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as (cite).  

2) The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.  

3) Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.

## Dataset 
Multi30k : WMT Multimodal Translation: de-en
  
## Bleu-Score
The model reaches a bleu score of 35.44 which is comparable to the state of the art performances of recent models such as 'Multi-Agent Dual Learning' which reaches ~40 as reported on this [leaderboard](https://paperswithcode.com/sota/machine-translation-on-wmt2016-english-german).


