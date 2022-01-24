# Graph Attention Networks
A TensorFlow 2 implementation of Graph Attention Networks for classification of nodes from the paper, [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., ICLR 2018).

This is my attempt at trying to understand and recreate the neural network from from the paper. You can find the official implementation here: https://github.com/PetarV-/GAT

## Requirements
- tensorflow 2
- networkx
- numpy
- scikit-learn

## Run

To train and test the network with the CORA dataset.

```bash
python train.py
```

## Cite

Please cite the original paper if you use this code in your own work:

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
