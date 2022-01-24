# Graph-based joint model with Nonignorable Missingness (GNM)
This is a Keras implementation of the GNM model in paper ’Graph-Based Semi-Supervised Learning with Nonignorable Nonresponses‘ by Fan Zhou et al (NeurIPS 2019).

## Acknowledgements
This GNM model supports the architecture of 

Graph Convolution Network (Thomas N. Kipf, Max Welling ICLR 2017), [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907),

Graph Attention Networks (Veličković *et al.*, ICLR 2018): [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

We build our pipeline based on [Keras Graph Attention Network](https://github.com/danielegrattarola/keras-gat) and [Keras Graph Convolution Network](https://github.com/tkipf/keras-gcn).

You should cite these papers if you use any of this code for your research:
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={Accepted as poster},
}

@article{velivckovic2017graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1710.10903},
  year={2017}
}
```

I copied the code in `utils.py` almost verbatim from [this repo by Thomas Kipf](https://github.com/tkipf/gcn) and add some new codes such as evaluation model prediction performance split training/validation/test data. 


## Disclaimer
I do not own any rights to the datasets distributed with this code, but they are publicly available at the following links:

- CORA: [https://relational.fit.cvut.cz/dataset/CORA](https://relational.fit.cvut.cz/dataset/CORA)
- PubMed: [https://catalog.data.gov/dataset/pubmed](https://catalog.data.gov/dataset/pubmed)
- CiteSeer: [http://csxstatic.ist.psu.edu/about/data](http://csxstatic.ist.psu.edu/about/data)


## Replicating experiments
To replicate the simulation results of the paper, simply run:
```sh
$ python sim_cora_GCN.py
```
and 
```sh
$ python sim_cora_GAT.py
```
for the Cora dataset with ‘lambda = 2’ or

```sh
$ python sim_citeseer_GAT.py
```
and 
```sh
$ python sim_citeseer_GCN.py
```
for the Citeseer dataset.

To replicate the simulation results of the paper, simply run:
```sh
$ python real_cora.py
```
