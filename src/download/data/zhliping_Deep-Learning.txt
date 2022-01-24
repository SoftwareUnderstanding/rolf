Attention-based Graph Neural Network
============
Related Works
------------
- GIN Paper Link: [https://arxiv.org/abs/1810.00826](https://arxiv.org/abs/1810.00826)
- GAT Paper Link: [https://arxiv.org/abs/1807.07984](https://arxiv.org/abs/1807.07984)
- GAT Pytorch Tutorial Link: [https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)


Library Needed
------------
- torch
- dgl
- numpy


Running Instructions
----------
- We have three dataset "cora", "citeseer" and "pubmed". To specify which dataset to use, use --dataset on the command line.
- To use GPU, use --gpu=0 on the command line.
- We have five models: 1(GCN), 2(GAT), 3(MLPGAT), 4(MLPGAT_1), 5(MLPGAT_average), 6(ENSEMBLE of 3,4,5). To select model, use --model on the command line.
- To change the hyperparameters of training, check the code in train.py for details.


Running Samples
----------
```bash
python train.py --dataset=cora --gpu=0 --model=1
```

```bash
python train.py --dataset=citeseer --gpu=0 --model=3
```

```bash
python train.py --dataset=pubmed --gpu=0 --model=5
```