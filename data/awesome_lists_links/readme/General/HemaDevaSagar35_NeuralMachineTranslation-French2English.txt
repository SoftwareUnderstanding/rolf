## Introduction

Few months back when trying to search for Subtitles for a French Movie, I got an idea to build a mini-version of Neural Machine Translation system for French - English and see how it feels to build one. Courses CS 224n : Natural Language Processing with Deep learning and Sequence models from Coursera helped a lot in understanding Sequence models, although there is a long way to go!

## Setup for experimentation

Knowing that my Laptop doesn't have great configuration to train deep neural networks, I planned my experimentation on GCP. FYI, for a first time user free credits worth 300$ will be given. Lot of articles are online which shows step-by-step procedure for setting-up a GCP instance powered with GPU. The article in the [link](https://medium.com/google-cloud/using-a-gpu-tensorflow-on-google-cloud-platform-1a2458f42b0) explains the steps in very sane manner. 

## Neural Machine Translation 

NMT model I attempted to build here belongs to the family of encoder-decoder models, but with addition of attention to learn alignment and translation jointly. With Pure encoder-decoder model the issue is that, like mentioned in the [paper](https://arxiv.org/pdf/1409.0473.pdf), the network's encoder tries to compress all the information into fixed length vector and then the decoder attempts to translate which makes it difficult for the overall network to cope up with long sentences. Wheareas the Model used here following the [paper](https://arxiv.org/pdf/1409.0473.pdf) encodes the input senctence into sequence of vectors and chooses a subset of these vectors adapatively through attention while decoding the translation.

Courses like Sequence models from Cousera and CS 224n are very helpful in understanding differences vetween RNN's, LSTM's, GRU's, Bidirectional units intuition on encoders, decoders, attention mechanism etc. NMT model built here for experimentation is a modified version of the network published in the [paper](https://arxiv.org/pdf/1409.0473.pdf). Shrinked version was built to reduce training time which makes it possible to conduct more experiments, given the budget was limited! Specifications of the architecture built can be seen in the python files. 

## Dataset for the task

Used the paralled corpora from french-english provided by ACL WMT'14, [link here](http://www.statmt.org/wmt14/translation-task.html). This dataset contains some 40M odd samples. Sampled approximately 140K samples for training and 2.3K for testing the model. 

## Training the model

The encoder part of the model is composed of Bidirectional LSTM, whereas the decoder part of the model is composed of unidirectional LSTM with 512 hidden units each. An input length 30 words is used, further expermentation should be done by increasing this, and a total vocabulary of size 30K is considered for both French and English individually. An embedding of size 100 dimension is used.

Trained the model for almost 5 days till 400 epochs. Used beam search to find a translation that approximately maximizes the conditional probability, [link to the paper](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), and obtained a Bleu score of 10.9 on the above sampled test set

## Some Examples
**example 1** : <br />
**French input** : comité préparatoire de la conférence des nations unies sur le commerce illicite des armes légères sous tous ses aspects <br />
**Actual English Translation** : preparatory committee for the united nations conference on the illicit trade in small arms and light weapons in all its aspects <br />
**Model's English Translation** : preparatory committee for the united nations conference on the illicit trade in small arms and light weapons in all its aspects <br />


**example 2** : <br />
**French input** : il est grand temps que la communauté internationale applique cette résolution <br />
**Actual English Translation** : it was high time that the international community implemented that resolution <br />
**Model's English Translation** : it is high time that the international community should be adopted by the resolution <br />

**example 3** : <br />
**French input** : conclusions concertées sur l'élimination de toutes les formes de discrimination et de violence à l'égard des petites filles <br />
**Actual English Translation** : agreed conclusions on the elimination of all forms of discrimination and violence against the girl child <br />
**Model's English Translation** : conclusions conclusions on the elimination of all forms of discrimination and violence against the young people. <br />

## Future work
* Visualize the model's attention on different words while translating a source language sample (language being French here)
* Play with the architecture and its hyperparameters like for example,  changing the dimensions and hidden states units of embeddings and LSTM's respectively, varying the input sentence length, Stacking more layers on the encoder and decoder etc.
* Explore [Attention is all you need] and implement a version of this for the task of translation

## References
* [Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014)](https://arxiv.org/pdf/1409.0473.pdf)
* [Sutskever, I., Vinyals, O. and Le, Q.V., 2014. Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112)](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [Papineni, K., Roukos, S., Ward, T. and Zhu, W.J., 2002, July. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics](https://www.aclweb.org/anthology/P02-1040)
* [https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention)
* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
* [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models?)
