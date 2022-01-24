# Apply aleatoric uncertainty to Sentiment Analysis using ELMO

This repository tries to reproduce the results for the sentiment analysis task as reported in the paper 
"Deep Contextualized Word Embeddings"

*Acknowledgements:* This work is based on the previous work reported in https://raw.githubusercontent.com/markmenezes11/COMPM091/ where 
they reproduce in Tensorflow the BCN as described in the paper "Learned in Translation: Contextualized Word Vectors" by McCann et al.

In this work we change the Cove+Glove word embeddings by the ELMO embeddings, also changing the final Maxout network as 
described in the ELMO's paper.

Structure of the report:
* datasets: this directory holds the datasets used. In our csase we only use the SST-5 from Socher et al. where we have 
Staforn Sentiment Analysis Tree Banck for movie reviews labeled in 5 categories from very negative to very positive 
(SST-5)
* Cove-ported: Cove original model ported to Keras. This is useful for reporting the results obtained in the "Learned in
 Translation". The file is compatible with Python3 and introduces a dependency to Keras.
* Cove-BCN: defines the Cove BiAttentive Classification Network, BCN, as implemented by Mark Menezes, but removing the 
InferSent part and other datasets but SST-5.
    * `eval.py` is the main script for running the evaluation using the BCN.
    * `datasets.py` loads the datasets for the given transfer task.
    * `sentence_encoders.py` uses the InferSent or CoVe model to generate sentence embeddings for each sentence.
    * `model.py` contains all of the TensorFlow code for the BCN model.
* ELMO-BCN: defines the adaptation of the original Cove-BCN as described in ELMO paper, including using ELMO embeddings 
and removing the Maxout LAyers.
    * `eval.py` is an adaptation of the original script for handling ELMO relative parameters
    * `sentence_encoders.py` holds the ELMO embedder, base on the implementation found in Tensorflow Hub.
    * `model.py` contains all of the TensorFlow code for the BCN model with the mentioned changes.

## Cove-BCN Run

You can download GloVe embeddings with:
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

See `python eval.py -h` before running the scripts properly, for details on what the parameters are, as you will have to set all of the paths and flags that you need. Also open the scripts in an editor if you want to see an example of what the parameters are set to by default.

Run the evaluation with:
```
python eval.py
```


## References

* [Mark Menezes, MEng Computer Science Undergraduate Final Year Individual Projct (COMPM091) at University College London](https://github.com/markmenezes11/COMPM091)
* [Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.](https://arxiv.org/pdf/1802.05365.pdf)
* [Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loïc, and Bordes, Antoine. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 670–680. Association for Computational Linguistics, 2017.](https://arxiv.org/pdf/1705.02364.pdf)
* [McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp. 6297–6308. Curran Associates, Inc., 2017.](https://arxiv.org/pdf/1708.00107.pdf)
* [Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543, 2014.](https://nlp.stanford.edu/pubs/glove.pdf)
* Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
* Pytorch. [online]. Available at: https://github.com/pytorch/pytorch.
* Chollet, Francois et al. Keras. [online]. Available at: https://github.com/keras-team/keras, 2015.
* Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
* Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.
* E. M. Voorhees and D. M. Tice. The TREC-8 question answering track evaluation. In TREC, volume 1999, page 82, 1999. 