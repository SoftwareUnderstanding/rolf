# COVID GAT

Covid Prediction PyTorch Graph Attention Network

This repo is a PyTorch implementation of a Graph Attention Network for the prediction of covid cases based on an adjacency matrix of geographical nodes with edges indicating commuting between nodes and edge features including population, proportion of population 60+ years old and previous covid deaths and cases.

Original PyTorch implementation created by, and forked from, [@Diego999](https://github.com/Diego999/pyGAT.git).

From the above Repo:

This is a pytorch implementation of the Graph Attention Network (GAT)
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

The repo has been forked initially from https://github.com/tkipf/pygcn. The official repository for the GAT (Tensorflow) is available in https://github.com/PetarV-/GAT. Therefore, if you make advantage of the pyGAT model in your research, please cite the following:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```
