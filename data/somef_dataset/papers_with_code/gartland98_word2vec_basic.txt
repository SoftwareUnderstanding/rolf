# word2vec_basic

Dataset

text8 corpus

How to run

python word2vec.py [mode] [negative_samples] [partition]

mode : "SG" for skipgram, "CBOW" for CBOW
partition : "part" if you want to train on a part of corpus (faster training but worse performance), 
             "full" if you want to train on full corpus (better performance but slower training)

Examples) 
python word2vec.py SG  full // SG trained by full corpus
python word2vec.py CBOW part // CBOW trained by part of corpus 1000000 words

You should adjust the other hyperparameters in the code file manually.
cited paper: Efficient Estimation of Word Representations in Vector Space https://arxiv.org/abs/1301.3781
