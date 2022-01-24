# TensorFlow seq2seq model

<p align="center">
  <img src="g3doc/files/seq2seq.png" width="900">
</p>


This is a TensorFlow 2.x implementation of the seq2seq model augmented with attention mechanism (Luong-style or Bahdanau-style) for neural machine translation. Follow this [guide](https://github.com/chao-ji/tf-seq2seq/blob/master/g3doc/Build_seq2seq_model.md) for a conceptual understanding about how seq2seq model works. 


## Data Preparation, Training, Evaluation, Attention Weights Visualization 
The implementation in this repo is designed to have the same command line interface as the [Transformer](https://github.com/chao-ji/tf-transformer) implementation. Follow that link for detailed instructions on data preparation, training, evaluation and attention weights visualization.

### Visualize Attention Weights 
Unlike [Transformer](https://github.com/chao-ji/tf-transformer), the seq2seq model augmented with attention mechanism involves only *target-to-source* attention. Shown below is the attention weights w.r.t each source token (English) when translating the target token (German) one at a time.

<p align="center">
  <img src="g3doc/files/alignment.png" width="900">
  English-to-German Translation 
</p>

# References
* Effective Approaches to Attention-based Neural Machine Translation, Luong *et al.* [https://arxiv.org/abs/1508.04025](https://arxiv.org/abs/1508.04025)
* Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau *et al.* [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

