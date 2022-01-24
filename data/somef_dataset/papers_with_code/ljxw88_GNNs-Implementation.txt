<div align="center">
<a href="" target="_blank">
   <img src="src/image/logo.png" alt="repo logo" style="width:10%">
</a>

# Playground of Graph Neural Networks

_Hey there! This will be my repo implementing/collecting GRAPH related DL models, with some tutorials and notes! **I will consistently update this repository**. Have fun!_

</div>


## Environment

We recommend using jupyterlab for running and debugging. Most models have the complex package dependencies, we strongly recommend using **Anaconda** for package management and deployment. You can use the following command to create a conda environment similar to ours:
```
conda env create -f src/environment_droplet.yml
```
***Note that** we maintain and update our environment droplet consistently, so do stick to the latest one :-)

## Models To-Do Lists

- `2021-10-22 GCN-PyTorch` Graph Convolutional Network (GCN) in PyTorch, similar to [[paper](https://arxiv.org/abs/1609.02907)]. 🚀


- `2021-10-27 DGI-PyTorch` Deep Graph Infomax (DGI) in PyTorch, similar to [[paper](https://arxiv.org/abs/1809.10341)]. ❌

- `2021-10-28 GAE-PyTorch` Graph Auto-Encoder (GAE) in PyTorch, similar to [[paper](https://arxiv.org/abs/1611.07308)]. ❌

## Reference Models

- `2021-10-22 GCN-tkipf` Graph Convolutional Networks in PyTorch (Thomas Kipf) [[repo](https://github.com/tkipf/pygcn)]. Check out the [[code folder](./GCN-tkipf)] with *Readme* in this repo for notes and tutorials.

## DataSets
- `2021-10-28 update` We decide to use Open Graph Benchmark (OGB) as the only benchmarking protal for this project: [[GitHub](https://github.com/snap-stanford/ogb)], [[HomePage](https://ogb.stanford.edu/)]. We are working on transfering the project dataset to the OGB dataset.

## Miscellaneous

None

## Cite
```
@software{Liu_Playground_of_Graph_2021,
author = {Liu, Jiaxu},
month = {10},
title = {{Playground of Graph Neural Networks}},
url = {https://github.com/ljxw88/GNNs-Implementation},
year = {2021}
}
```

## Liscense
GPL-3.0
