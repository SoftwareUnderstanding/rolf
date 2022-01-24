# **XLNet**
A Julia based implementation of **XLNet: A Generalized Autoregressive Pretraining for LU. (Flux and JuliaText)**
- #### [Original Implementation - TensorFlow: *zihangdai/xlnet*](https://github.com/zihangdai/xlnet)
- #### [HuggingFace Implementation - PyTorch: *huggingface/transformers/**src/transformers/models/xlnet***](https://github.com/huggingface/transformers/tree/master/src/transformers/models/xlnet)
# 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SambhawDrag.github.io/XLNet.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SambhawDrag.github.io/XLNet.jl/dev)
[![Build Status](https://github.com/SambhawDrag/XLNet.jl/workflows/CI/badge.svg)](https://github.com/SambhawDrag/XLNet.jl/actions)
[![Build Status](https://travis-ci.com/SambhawDrag/XLNet.jl.svg?branch=master)](https://travis-ci.com/SambhawDrag/XLNet.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/SambhawDrag/XLNet.jl?svg=true)](https://ci.appveyor.com/project/SambhawDrag/XLNet-jl)

- **License** : [MIT License](https://github.com/SambhawDrag/XLNet.jl/blob/master/LICENSE)
## **What is XLNet?**
XLNet is an generalized autoregressive pretraining for language understanding. The [XLNet paper](https://arxiv.org/abs/1906.08237) combines recent advances in NLP with innovative choices in how the language modelling problem is approached. When trained on a very large NLP corpus, the model achieves state-of-the-art performance for the standard NLP tasks that comprise the GLUE benchmark.

XLNet is an *auto-regressive* language model which outputs the **joint probability** of a sequence of tokens based on the [**Transformer**](https://arxiv.org/abs/1901.02860) architecture with recurrence. Its training objective calculates the probability of a word token conditioned on **all permutations of word tokens in a sentence, as opposed to just those to the left or just those to the right of the target token**.
#
## **What makes XLNet so special?**
XLNet was proposed by researchers at Google Inc. in 2019. Since,
- The autoregressive language model (e.g.GPT-2) is only trained to encode a unidirectional context and not effective at modeling deep bidirectional contexts, and 
- Autoencoding (e.g.BERT) suffers from the pre-train fine-tune discrepancy due to masking, 
XLNet borrows ideas from the two types of objectives while avoiding their limitations.

It is a new objective called **Permutation Language Modeling**. By using a permutation operation during training time, bidirectional context information can be captured and makes it a generalized order-aware autoregressive language model. No masking is required and thus the dependency between the BERT [MASK] tokens is maintained. Besides, XLNet introduces a two-stream self-attention to solve the problem that standard parameterization will reduce the model to bag-of-words. 

Additionally, XLNet employs **Transformer-XL** as the backbone model, exhibiting excellent performance for language tasks involving long context.

Two versions of the XLNet model have been released, i.e. 
1. [**XLNet-Large, Cased**](https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip) : 24-layer, 1024-hidden, 16-heads
2. [**XLNet-Base, Cased**](https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip) : 12-layer, 768-hidden, 12-heads
and, they include similar settings of the corresponding BERT. 

>**XLNet _(Paper Abstract)_: **Empirically, XLNet outperforms BERT on 20 tasks and achieves state-of-the-art results on 18 tasks.** 

---
### **1. XLNet benefits from Auto-Regression and Auto-Encoding models**
> Add here 1
---
### **2. Permutation Language Modelling**

<p align="center">
<img src="doc_img/PLM.png" width="100%" height="100%">
</p>

Specifically, for a sequence _**X{...}**_ of length _**T**_, there are _**T**_ different orders to perform a valid autoregressive factorization! Intuitively, if model parameters are shared across all factorization orders, in expectation, the model will learn to gather information from all positions on both sides. Let, _**P<sub>T</sub>**_ be the set of all possible permutations of a sequence **[1,2,…, T]** and use _**z<sub>t</sub>**_ and _**z<sub><t</sub>**_ to denote the _t-th element_ and the _first t−1_ elements of a permutation, _**p ∈ P<sub>T</sub>**_. 
Then the permutation language modeling objective can be expressed as follows:
For instance, assume we have a input sequence {I love my dog}.

In the upper left plot of the above figure, when we have a factorization order: **{3, 2, 4, 1}**, the probability of sequence can be expressed as follows:

- **For the third token:** {my}, it cannot use the information of all other tokens, so only one arrow from the starting token points to the third token in the plot.

In the upper right plot of the figure, when we have a factorization order: **{2, 4, 3, 1}**, the probability of sequence can be expressed as follows:

- **Here, for the third token:** {my}, it can use the information of the second and fourth tokens because it places after these two tokens in the factorization order. Correspondingly, it cannot use the information of the first token. So in the plot, in addition to the arrow from the starting token, there are arrows from the second and fourth tokens pointing to the third token. The rest two plots in the figure have the same interpretation.

During training, for a fixed factorization order, XLNet is a unidirectional language model based on the transformer decoder, which performs normal model training. But different factorization order makes the model see different order of words when traversing sentences. **In this way, although the model is unidirectional, it can also learn the bidirectional information of the sentence.**

> It is noteworthy that the sequence order is not actually shuffled but only attention masks are changed to reflect factorization order. With PLM, XLNet can model bidirectional context and the dependency within each token of the sequence.

---
### **3. Two-Stream Self-Attention with Target-Aware Representation**
> Add here 2
---
#

## **Work Checkpoints**
###### **- May not be updated -**
#### Dated : 04-06-2021  (**May not be updated**)
- [ ] Convert pre-train weights to bson
- [ ] Create tokenizer : sentence-piece 
- [ ] Add as xlnet_tokenizer.jl
- [ ] Transformer-XL encoder-decoder base with features essential to XLNet
- [ ] ...

## **Status**
In progress

## **References**
1. [**XLNet: Generalized Autoregressive Pretraining for Language Understanding** - _arxiv.org_](https://arxiv.org/abs/1906.08237)
2. [**Understanding XLNet** - _Borealis AI_](https://www.borealisai.com/en/blog/understanding-xlnet/)
3. [**Understanding Language using XLNet with autoregressive pre-training** - _medium.com_](https://medium.com/@zxiao2015/understanding-language-using-xlnet-with-autoregressive-pre-training-9c86e5bea443)
4. [**Sentence-Piece Subword Tokenizer** - _Google_](https://github.com/google/sentencepiece)
5. [**Permutation Language Modelling** - _LMU Munich_](https://compstat-lmu.github.io/seminar_nlp_ss20/transfer-learning-for-nlp-ii.html#permutation-language-modelingplm)
