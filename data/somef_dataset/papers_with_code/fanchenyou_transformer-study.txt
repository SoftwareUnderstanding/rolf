## Several Transformer network variants tutorials

#### 1. transformer_encoder, [paper](https://arxiv.org/pdf/1706.03762.pdf), [src](https://pytorch.org/tutorials/beginner/transformer_tutorial.html), [tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    * Use Pytorch nn.transformer package to build an encoder for language prediction
    * PyTorch 1.2 + TorchText
   
#### 2 & 2.1. transformer_xl_from_scratch, [src](https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/)
    * 2. simple toy example showing the idea of Transformer-XL which uses additional memory to encode history    
    * 2.1 Build Transformer-XL + MultiAttention heads
    * Show how to use previous hidden states to achieve "Recurrence Mechanism"
      - the output of the previous hidden layer of that segment
      - the output of the previous hidden layer from the previous segment
    * Show how to use relative positional encoding to incorporate position information

#### 3. transformer_xl full release, [src](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch), [tutorial](https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924)
  ![Network](/pics/pic1.png)
  
    * Complete implementation of Transformer-XL
    
#### 4. xlnet, [paper](https://arxiv.org/pdf/1906.08237.pdf), [src](https://github.com/graykode/xlnet-Pytorch), [tutorial](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335)
    * An excellent tutorial version of XLNet from above link
    * Add more comments for understanding
    * Requirements: Python 3 + Pytorch v1.2 
    * TODO: Add GPU support

#### 5. Bert from scratch, [paper](https://arxiv.org/abs/1810.04805), [src](https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model), [tutorial](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
    * Build Bert - Bidirectional Transformer
    * The task is two-fold, see paper section 3.1
        1) to predict the second part of a sentence (Next Sentence Prediction)
        2) to predict the masked words of a sentence (Masked LM)
    * step 1: generate vocabulary file "vocab.small" in ./data
    * step 2: train the network
    * See transformer_bert_from_scratch_5.py for more details.

#### 6. Bert from Pytorch Official Implementation, [paper](https://arxiv.org/abs/1810.04805), [src](https://github.com/huggingface/transformers)
    * Build Bert - Bidirectional Transformer
    * Utilize official Pytorch API to implement the interface of using existing code and pre-trained model
    * pip install transformers tb-nightly 


#### 7. ALBERT, A Lite BERT, [paper](https://arxiv.org/pdf/1909.11942.pdf), [src](https://github.com/graykode/ALBERT-Pytorch), [tutorial](https://medium.com/@lessw/meet-albert-a-new-lite-bert-from-google-toyota-with-state-of-the-art-nlp-performance-and-18x-df8f7b58fa28)
    * A Lite BERT which reduces BERT params to ~20%
    * Decouple word embedding size with hidden size by using two word projection matrices 
       - parameters are reduced from O(V*H) to O(V*E + E*H) s.t. E << H
    * Cross-layer parameter sharing
       - the default decision for ALBERT is to share all parameters across layers (see paper section 3.1 !!)
    * Sentence Order Prediction
       - NSP (Next Sentence Prediction) in BERT is not effective (as the association of two sents in a doc is not strong)
       - Inter-sentence coherence is strong: 
         the positive case is the two sentences are in proper order; 
         the negative case is the two sentences in swapped order.


### Requirements
Python = 2.7 and 3.6

PyTorch = 1.2+ [[here]](https://pytorch.org/) for both python versions

GPU training with 4G+ memory, testing with 1G+ memory.
