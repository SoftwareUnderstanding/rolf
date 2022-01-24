# EUMT

English-Ukrainian bidirectional neural machine translator, based on [fastText](https://fasttext.cc/docs/en/support.html) word embeddings (*sisg-* model [1]) and default Transformer architecture [2] of the [OpenNMT framework](https://opennmt.net/).

The following [OPUS datasets](https://opus.nlpl.eu/) [3] were used for training:

- [WikiMatrix](https://opus.nlpl.eu/WikiMatrix-v1.php) [4];
- [XLEnt](https://opus.nlpl.eu/XLEnt-v1.php) [5];
- [Tatoeba](https://opus.nlpl.eu/Tatoeba-v2021-03-10.php) [3];
- [QED](https://opus.nlpl.eu/QED-v2.0a.php) [6].

> Launch translator:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EugeneSel/EUMT/HEAD?urlpath=%2Fvoila%2Frender%2Fweb_app.ipynb)


**Check out [my article](http://scinews.kpi.ua/article/view/236939), related to this project.**

## References

1. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). [Enriching word vectors with subword information](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00051?source=post_page---------------------------). Transactions of the Association for Computational Linguistics, 5, 135-146.
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf%EF%BC%89%E6%8F%8F%E8%BF%B0%E4%BA%86%E8%BF%99%E6%A0%B7%E5%81%9A%E7%9A%84%E5%8E%9F%E5%9B%A0%E3%80%82). arXiv preprint arXiv:1706.03762.
3. Jörg Tiedemann, 2012, [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf). In *Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC'2012)*.
4. Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong and Paco Guzman, [WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia](https://arxiv.org/abs/1907.05791), arXiv, July 11 2019.
5. Ahmed El-Kishky, Adi Renduchintala, James Cross, Francisco Guzmán and Philipp Koehn, [XLEnt: Mining Cross-lingual Entities with Lexical-Semantic-Phonetic Word Alignment](http://data.statmt.org/xlent/elkishky_XLEnt.pdf), Online preprint, 2021.
6. A. Abdelali, F. Guzman, H. Sajjad and S. Vogel, "[The AMARA Corpus: Building parallel language resources for the educational domain](https://www.aclweb.org/anthology/L14-1675/)", The Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC'14). Reykjavik, Iceland, 2014. Pp. 1856-1862. Isbn. 978-2-9517408-8-4.
