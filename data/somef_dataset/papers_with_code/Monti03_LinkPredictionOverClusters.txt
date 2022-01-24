# LinkPredictionOverClusters
## Approach
In this repository, we have developed different techniques that try to exploit the clusters obtained from a graph to improve link prediction results. The proposed models are the following ones:
- couple models: for each couple of clusters train a GAE model and predict the test edges between nodes inside a single cluster as avg of the predictions of the models trained over such cluster
- single models with FC: for each cluster, we train a GAE model and for each couple of clusters we train an FC that is used to map the embeddings of each node over a common dimension with the nodes of the other cluster
- shared model: a model where one of the two convolutional layers of the GAE model is shared among the different clusters
- shared model with adversarial loss: to improve the precedent model we tried to use an adversary loss to let the embeddings in output from the shared layer be independent of the cluster from which they come
- single models with adversarial loss: in this case, we train one GAE model for each cluster and use an adversary loss to let the embeddings in output be independent of the cluster from which they come
## Results for 3 clusters on Facebook dataset
| Model                               | F1 Score      | Training Time (s) |
| -------------                       | ------------- | -------------     |
| baseline model                      |0.9206         |  727              |
| couple models                       | 0.9268        | 1646              |
| single models with FC               |  0.8940       |517                |
| shared first                        |  0.8977       | 719|
| shared last                         |  0.8900       | 511|
| shared last with adversarial loss   | 0.8900        | 1051|
|single models with adversarial loss  | 0.8287        | 993|

## Instruction
- Shared Model: ``python3 train_shared_model.py``
- Shared Model with Adversary Loss: ``python3 train_shared_model.py --adv``
- Single Models with Adversary Loss: ``python3 train_one_per_clust_adv_loss.py``
- Single Models with FC: ``python3 train_single_models_and_fc_between.py --use_fc``
- Couples: ``python3 train_couples.py``

Other usefull parameters are:
- ``--test``: to run a quick test with a few epochs
- ``--dataset=`` to chose the dataset (from pubmed, [amazon electronics](https://github.com/spindro/AP-GCN/tree/master/data) and [facebook](http://snap.stanford.edu/data/facebook-large-page-page-network.html))
- ``--n_clusters=`` to chose the number of cluster to consider

## Dependencies
``pip3 install scipy numpy tensorflow pandas networkx seaborn sklearn matplotlib``
