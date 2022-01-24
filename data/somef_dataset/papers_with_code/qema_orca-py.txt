# PyORCA: ORCA orbit Counting Python Wrapper

This is a wrapper to the C++ ORCA algorithm, published in "A combinatorial approach to graphlet counting", Bioinformatics 2014.

## Use as Feature
Different from ORCA's original purpose, a main use case of this library is to use the orbit counts as features in machine learning for graph-structured data.
Graph neural networks (GNNs) achieve state-of-the-art in many graph learning domains, such as [biology](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770), [chemistry][http://papers.nips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation], [recommender systems][https://arxiv.org/pdf/1806.01973.pdf], among many others.
Some of the example architectures include [GCN](https://arxiv.org/abs/1609.02907), [GraphSAGE](https://arxiv.org/abs/1706.02216) and [GAT](https://arxiv.org/abs/1710.10903).
Pytorch Geometric and DGL are useful libraries for GNNs.

In addition to the Python version of the C++ ORCA interface, this repo also provide python interface to easily create feature vectors for nodes based on orbit counts. They can be used as utilities to supply additional input features for nodes in a GNN.
