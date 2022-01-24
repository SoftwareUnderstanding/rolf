# GAT_pyG

this is a Graph Attention Networks(Pytorch_geometric version).

## How to Run
```bash
python train.py
```

If you need to know how to set the parameters, 
```bash
python train.py --help
```
(The parameter names of the command line arguments are the same as the ones listed in original paper(https://arxiv.org/abs/1710.10903))

## Guide to experimental replication
Experimental results described in original paper(https://arxiv.org/abs/1710.10903) for each dataset could be reproduced by setting up the following.
| dataset | n_class | weight_decay |
|:---:|:---:|:---:|
| Cora | 7 | 5e-4 |
| CiteSeer | 6 | 5e-4 |
| PubMed | 3 | 1e-3 |
