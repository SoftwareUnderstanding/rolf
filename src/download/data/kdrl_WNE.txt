SGNS-WNE : The word-like n-gram embedding version of skip-gram model with negative sampling
=======================================================

**SGNS-WNE** is an open source implementation of our framework to learn distributed representation of words by embedding word-like character n-grams, described in the following papers:
- **Word-like character n-gram embedding** [http://aclweb.org/anthology/W18-6120](http://aclweb.org/anthology/W18-6120)

## Requirements & Environment

- Linux(Tested with CentOS Linux release 7.4.1708)
- gcc(>=5)
- hdf5
- Python 3
- NumPy
- Pandas
- h5py
- scikit-learn
- tqdm
- [cmdline](https://github.com/tanakh/cmdline/blob/master/cmdline.h) : Download `cmdline.h` and place it in `2_count_ngram_frequency/`, `4_count_expected_word_frequenct/` and `5_SGNS_WNE/`

## Contents

* `1_preprocess/` : Pre-processing corpus. Sentences are concatenated and white spaces are replaces with another character for visualization.
* `2_count_ngram_frequency/` : Count n-grams frequency. In this implementation, we use lossy counting algorithm.
* `3_logistic_regression/` : Probabilistic predictor for word boundary.
* `4_count_expected_word_frequenct/` : Count expected word frequency (ewf) of word-like n-grams.
* `5_SGNS_WNE/` : Compute distributed representations of word-like n-grams via skip-gram model with negative sampling.

```
.
├── 1_preprocess
│   └── main.py
├── 2_count_ngram_frequency
│   ├── cmdline.h
│   ├── lossycounting.cpp
│   ├── lossycounting.h
│   ├── main.cpp
│   ├── makefile
│   └── run.sh
├── 3_logistic_regression
│   └── main.py
├── 4_count_expected_word_frequency
│   ├── cmdline.h
│   ├── counting_word.cpp
│   ├── counting_word.h
│   ├── main.cpp
│   ├── makefile
│   └── run.sh
├── 5_SGNS_WNE
│   ├── cheaprand.h
│   ├── cmdline.h
│   ├── main.cpp
│   ├── makefile
│   ├── run.sh
│   ├── skipgram.cpp
│   └── skipgram.h
└── README.md
```

# Submodules & Dependencies

* [word2vec - Google Codes](https://code.google.com/archive/p/word2vec/)
* [oshikiri/w2v-sembei - GitHub](https://github.com/oshikiri/w2v-sembei)
* [tanakh/cmdline - GitHub](https://github.com/tanakh/cmdline)

The majority of C++ code which is used for computing representations for n-grams with SGNS is taken from **[word2vec - Google Codes](https://code.google.com/archive/p/word2vec/)**[1] and **[w2v-sembei](https://github.com/oshikiri/w2v-sembei)**[2].

## References

1. Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. In Proceedings of ICLR2013. [[pdf](https://arxiv.org/pdf/1301.3781.pdf), [code](https://code.google.com/archive/p/word2vec/)]
2. Oshikiri, T. (2017). **Segmentation-Free Word Embedding for Unsegmented Languages**. In Proceedings of EMNLP2017. [[pdf](http://aclweb.org/anthology/D17-1080)]
3. Kudo, T., Yamamoto, K., & Matsumoto, Y. (2004). **Applying Conditional Random Fields to Japanese Morphological Analysis**. In Proceedings of EMNLP2004. [[pdf](http://www.aclweb.org/anthology/W04-3230)]
4. **MeCab: Yet Another Part-of-Speech and Morphological Analyzer**. [[code](http://taku910.github.io/mecab/)]
