# GAT modified
Graph Attention Networks (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

![](./img/tensorboard.png)

## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` store model checkpoint);
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);


## Sparse version
An experimental sparse version is also available, working only when the batch size is equal to 1.
The sparse model may be found at `models/sp_gat.py`.

You may execute a full training run of the sparse model on Cora through `execute.py`.

## Command line argument
```
optional arguments:
  -h, --help            show this help message and exit
  -s, --sparse          use sparse operation to reduce memory consumption
  --epochs EPOCHS       number of epochs
  --lr LR               learning rate
  --patience PATIENCE   for early stopping
  --l2_coef L2_COEF     l2 regularization coefficient
  --hid_units HID_UNITS [HID_UNITS ...]
                        numbers of hidden units per each attention head in
                        each layer
  --n_heads N_HEADS [N_HEADS ...]
                        number of attention head
  --residual            use residual connections
  --attention_drop ATTENTION_DROP
                        dropout probability for attention layer
  --edge_attr_directory EDGE_ATTR_DIRECTORY
                        directory storing all edge attribute (.npz file) which
                        stores the sparse adjacency matrix
  --node_features_path NODE_FEATURES_PATH
                        csv file path for the node features
  --label_path LABEL_PATH
                        csv file path for the ground truth label
  --log_directory LOG_DIRECTORY
                        directory for logging to tensorboard
  --train_ratio TRAIN_RATIO
                        ratio of data used for training (the rest will be used
                        for testing)
```

## data preparation
### edge_attr_directory
The directory that contains multiple .npz file. 
- Each .npz file stores the scipy sparse matrix (N, N) - the adjacency matrix of the edge attribute.
- reference: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html)

### node_features_path
a csv file containing the node attribute.
- The first row contains the features name
- should have the same node ordering as the edge attribute adjacency matrix

### label_path
a csv file containing the node label.
- The first row must be "id, label"
- id corresponds to the node id (zero indexing)
- label can be any string

## example (train)

```bash
$ git clone https://github.com/handasontam/GAT-with-edgewise-attention.git

$ cd data

$ curl https://transfer.sh/11fhgc/eth.tar.gz -o eth.tar.gz  # md5: 62aef8b070d7be703152419f16e830d1

$ tar -zxvf eth.tar.gz

$ cd ../

$ python execute.py \
--sparse \
--epochs 100000 \
--lr 0.008 \
--patience 50 \
--l2_coef 0.005 \
--hid_units 5 \
--n_heads 2 1 \
--residual \
--attention_drop 0.0 \
--edge_attr ./data/eth/edges \
--node_features_path ./data/eth/node_features.csv \
--log_directory /tmp/tensorboard \
--label_path ./data/eth/label.csv

$ tensorboard --logdir=/tmp/tensorboard  # to run tensorboard
```
Once TensorBoard is running, navigate your web browser to localhost:6006 to view the TensorBoard

## example (load model)

``` bash
$ curl https://transfer.sh/iMacq/pre_trained.tar.gz -o pre_trained.tar.gz  # md5: 041de9eb6e7dcd4ca74267c30a58ad70

$ tar -zxvf pre_trained.tar.gz

$ python load_model.py \
--sparse \
--hid_units 5 \
--n_heads 2 1 \
--residual \
--edge_attr./data/eth/edges \
--node_features_path ./data/eth/node_features.csv \
--label_path ./data/eth/label.csv \
--train_ratio 0.8 \
--model_path ./pre_trained/mod_test.ckpt

$ # should print: Test loss: 0.579380989074707 ; Test accuracy: 0.86021488904953

```

## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`
- `pandas==0.23.4`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Reference
If you make advantage of the GAT model in your research, please cite the following in your manuscript:

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

## License
MIT
=======
