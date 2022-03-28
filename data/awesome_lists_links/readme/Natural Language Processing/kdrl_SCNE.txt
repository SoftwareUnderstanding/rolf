# SCNE

**SCNE** is an open source implementation of the paper "[Segmentation-free compositional n-gram embedding](https://www.aclweb.org/anthology/N19-1324)". NAACL-HLT2019.

If you find this code useful for your research, please cite the following paper in your publication:

    @inproceedings{kim-etal-2019-segmentation,
        title = "Segmentation-free compositional $n$-gram embedding",
        author = "Kim, Geewook  and
          Fukui, Kazuki  and
          Shimodaira, Hidetoshi",
        booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
        month = jun,
        year = "2019",
        address = "Minneapolis, Minnesota",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/N19-1324",
        pages = "3207--3215",
        abstract = "We propose a new type of representation learning method that models words, phrases and sentences seamlessly. Our method does not depend on word segmentation and any human-annotated resources (e.g., word dictionaries), yet it is very effective for noisy corpora written in unsegmented languages such as Chinese and Japanese. The main idea of our method is to ignore word boundaries completely (i.e., segmentation-free), and construct representations for all character $n$-grams in a raw corpus with embeddings of compositional sub-$n$-grams. Although the idea is simple, our experiments on various benchmarks and real-world datasets show the efficacy of our proposal.",
    }

## Requirements & Pretest Environment

- Linux (Demo program tested on CentOS Linux release 7.4.1708)
- gcc (>=5)
- Python 3
- NumPy
- Pandas
- SciPy
- scikit-learn

## How to Use

1. Initialize the submodules using
```
git submodule init
git submodule update
```
2. Compile with `make`
3. Prepare a training corpus. As a preprocessing, we replaced all whitespaces with '␣' (U+2423 Open Box Unicode Character)
4. Train embeddings (See `train.sh`)
5. Load the learned compositional n-gram embeddings and Use it (See `scne.py`)

## Submodules & Dependencies

The code is based on:

* [word2vec - Google Codes](https://code.google.com/archive/p/word2vec/)
* [oshikiri/w2v-sembei - GitHub](https://github.com/oshikiri/w2v-sembei)
* [jarro2783/cxxopts - GitHub](https://github.com/jarro2783/cxxopts)

## References

1. Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. In Proceedings of ICLR2013. [[pdf](https://arxiv.org/pdf/1301.3781.pdf), [code](https://code.google.com/archive/p/word2vec/)]
2. Oshikiri, T. (2017). **Segmentation-Free Word Embedding for Unsegmented Languages**. In Proceedings of EMNLP2017. [[pdf](http://aclweb.org/anthology/D17-1081)]
3. Sakaizawa, Y., & Komachi, M. (2017).  **Construction of a japanese word similarity dataset**. CoRR, abs/1703.05916. [[pdf](https://arxiv.org/abs/1703.05916), [data](https://github.com/tmu-nlp/JapaneseWordSimilarityDataset)]
