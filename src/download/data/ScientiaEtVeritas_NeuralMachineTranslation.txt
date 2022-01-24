# Attentional	Neural Machine	Translation
## Scope
Standard attentional-based NMT framework (Bahdanau et al. 2014)
- Data Preparer
- (Bidirectional) GRU/LSTM-based encoder
- Global Attention Layers: MLP/Dot Product (Luong et al. 2015)
- GRU/LSTM-based decoder
- Beam Search
- Teacher Forcing

## Requirements
- Data Preparation: Convert words to index prior to one-hot encoding for
training, validation and test sets.
- Training: Train the NMT architecture.
- Decoding/Testing: Realize the beam search to translate sentences from
German to English.

## Evaluation
Perplexity and BLEU scores on validation/test sets.

## Papers
- Effective Approaches to Attention-based Neural Machine Translation (2015, Luong et al., https://arxiv.org/pdf/1508.04025.pdf)
- Neural machine translation by jointly learning to align and translate (2014, Bahdanau et al., https://arxiv.org/pdf/1409.0473.pdf)
- Neural Machine Translation and Sequence-to-sequence Models: A Tutorial (2017, Neubig, https://arxiv.org/pdf/1703.01619.pdf)

## BLEU calculation
We used for the BLEU calculation the multi_bleu.pl script from the moses project: 
https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
