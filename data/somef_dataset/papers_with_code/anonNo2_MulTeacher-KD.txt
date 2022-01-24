# DARTS: Differentiable Architecture Search

Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)]

## Requirements

- python 3
- pytorch >= 0.4.1
- graphviz
    - First install using `apt install` and then `pip install`.
- numpy
- tensorboardX
- transformers

## Run example

- Augment

```shell
# genotype from search results
python3 augment.py --name sst --dataset sst --batch_size 256 --epochs 250 --limit 128 --n_classes 2 --genotype "Genotype(normal=[[('conv_3x3', 0)], [('highway', 0)], [('conv_3x3', 2)], [('conv_5x5', 1)], [('conv_3x3', 3)], [('conv_3x3', 1)]], normal_concat=range(1, 7), reduce=[], reduce_concat=range(1, 7))"
```

### Cautions

It is well-known problem that the larger batch size causes the lower generalization.
Note that although the [linear scaling rule](https://arxiv.org/pdf/1706.02677) prevents this problem somewhat, the generalization still could be bad.

Furthermore, we do not know about the scalability of DARTS, where larger batch size could be more harmful.
So, please pay attention to the hyperparameters when using multi-gpu.

## Reference

https://github.com/quark0/darts (official implementation)

### Main differences to reference code

- Supporting pytorch >= 0.4
- Supporting multi-gpu
- Code that is easy to read and commented.
- Implemenation of architect
    - Original implementation is very slow in pytorch >= 0.4.
- Tested on FashionMNIST / MNIST
- Tensorboard
- No RNN

and so on.
