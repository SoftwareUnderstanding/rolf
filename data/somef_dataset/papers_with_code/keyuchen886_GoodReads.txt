# GoodReads
This project will build a recommendation system for GoodReads. The dataset is available on https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home In our work, we only use the graphic subdata as it's comparitively small.

Currently, we have done the following:

1. Use a singular value decomposition based model called matrix factorization (https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) to create a baseline model. We use keras functional api to implement gradient descent, and our model includes bias and regularization terms for both book and users.

2. We use word2vec (skip-gram) schema (https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) to build user and book embeddings (100 dimension for each) in keras. For each user, we use one of his rated book as predictor and try to predict the other books he read. And we follow this fashion for each of his readed book. Hence, we consider the set of users' rated book is 'sentence' in the original paper. The book vector shows high correlation with reality: for example, the most similiar book of a japanese manga are also japense manga.

3. We use a deep and wide neural network architecture (https://arxiv.org/abs/1606.07792) which includes interaction between user and book to predict the rating score.

4. Our proposed model improve the mse from 2.6 to 1.2.


Next to do:
1. user part of speech tag to extract all adjectives in reviews
2. use topic modeling to see patterns exist in different books and detect whether there's group of semantics
