# Attention
## An Implementation of ''Attention Is All You Need'' 
### ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) ,(Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). This implementation is done in keras with tensorflow.
The paper presented a novel sequence to sequence framework that engaged the self-attention mechanism with feed forward network instead of Recurrent network structure, and achieve the state-of-the-art performance on WMT 2014 English-to-German translation task. (2017/06/12)
#### The architecture is named transformer and comprises of encoder and decoder with 6 stacks each.
##### Each encoder layer contain 2 sublayers with;
  * Multihead Self Attention Layer
  * Feed Forward Layer
##### Also the decoder layer has 3;
  * Feed Forward layer
  * Encoder to Decoder layer
  * Self Attention Layer.
##### Parameter settings:

      batch_size=64
      d_inner_hid=1024
      d_k=64
      d_v=64
      d_model=512
      d_word_vec=512
      dropout=0.1
      embs_share_weight=False
      n_head=8
      n_layers=6
      n_warmup_steps=4000
      proj_share_weight=True
 
 
  

## Requirements
  1. Python 3
  2. Numpy
  3. Tensorflow
  4. Keras
  
 ## 
