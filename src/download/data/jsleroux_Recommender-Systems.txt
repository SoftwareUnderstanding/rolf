# Deep Learning Recommender Systems
The objective of this notebook (project) is to test multiple recommendation systems for the movielens dataset.  I wanted to learn more on the following subjects:
- Pytorch: Learn more about this framework and implement some deep learning models
- Skorch: Get rid of the infamous Pytorch training loop.
- scikit-surprise: Learn how the framework works to obtain some benchmarks to compare my implementations

The dataset that was used is MovieLens 1m: http://files.grouplens.org/datasets/movielens/ml-1m-README.txt . The main reason I used this dataset is because it's the largest that contained side information.

I tried to implement a simple version of the following papers:  
1- **Matrix Factorization techniques with SGD learning** - https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

2- **Wide & Deep for recommender systems** - https://arxiv.org/pdf/1606.07792.pdf:
The wide part of the model are manual interactions. Those interactions have been built using the patsy package.
More information on feature crosses: https://datascience.stackexchange.com/questions/57435/how-is-the-cross-product-transformation-defined-for-binary-features

3- **Neural Collaborative Filtering** - https://arxiv.org/pdf/1708.05031.pdf
