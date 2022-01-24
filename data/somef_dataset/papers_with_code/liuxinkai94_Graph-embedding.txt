# Graph-embedding-master

# Method


|   Model   | Paper                                                                                                                      | Note                                                                                        |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| DeepWalk  | [KDD 2014][DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | [【Graph Embedding】DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812)  |
|   LINE    | [WWW 2015][LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)                          | [【Graph Embedding】LINE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56478167)      |
| Node2Vec  | [KDD 2016][node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) | [【Graph Embedding】Node2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56542707)  |
|   SDNE    | [KDD 2016][Structural Deep Network Embedding](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)               | [【Graph Embedding】SDNE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56637181)      |
| Struc2Vec | [KDD 2017][struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/pdf/1704.03165.pdf)        | [【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56733145) |


# How to run examples
1. clone the repo and make sure you have installed `tensorflow` or `tensorflow-gpu` on your local machine. 
2. run following commands
```bash
python setup.py install
cd examples
python deepwalk_wiki.py
```
## DeepWalk

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])# Read graph

model = DeepWalk(G,walk_length=10,num_walks=80,workers=1)#init model
model.train(window_size=5,iter=3)# train model
embeddings = model.get_embeddings()# get embedding vectors
```

## LINE

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])#read graph

model = LINE(G,embedding_size=128,order='second') #init model,order can be ['first','second','all']
model.train(batch_size=1024,epochs=50,verbose=2)# train model
embeddings = model.get_embeddings()# get embedding vectors
```
## Node2Vec
```python
G=nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                        create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])#read graph

model = Node2Vec(G, walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 1)#init model
model.train(window_size = 5, iter = 3)# train model
embeddings = model.get_embeddings()# get embedding vectors
```
## SDNE

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])#read graph

model = SDNE(G,hidden_size=[256,128]) #init model
model.train(batch_size=3000,epochs=40,verbose=2)# train model
embeddings = model.get_embeddings()# get embedding vectors
```

## Struc2Vec


```python
G = nx.read_edgelist('../data/flight/brazil-airports.edgelist',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])#read graph

model = model = Struc2Vec(G, 10, 80, workers=4, verbose=40, ) #init model
model.train(window_size = 5, iter = 3)# train model
embeddings = model.get_embeddings()# get embedding vectors
```
# seedNE
Tensorflow implementation of Self-Paced Network Embedding
## introduction
A tensorflow re-implementation of Self-Paced Network Embedding,use random walk to get positive pair
## Requirement
python 3.6, tensorflow 1.12.0
## Usage
To run the codes, use:
python seedne_new.py
## Result
on dataset cora,F1-micro:0.78,F1-macro:0.77
see SeedNE_new-Thu-May-16-10-22-56-2019-log.txt or result.png
# HARP
Code for the AAAI 2018 paper "HARP: Hierarchical Representation Learning for Networks".
HARP is a meta-strategy to improve several state-of-the-art network embedding algorithms, such as *DeepWalk*, *LINE* and *Node2vec*.

You can read the preprint of our paper on [Arxiv](https://arxiv.org/abs/1706.07845).

This code run with Python 2.

# Installation

The following Python packages are required to install HARP.

[magicgraph](https://github.com/phanein/magic-graph) is a library for processing graph data.
To install, run the following commands:

	git clone https://github.com/phanein/magic-graph.git
	cd magic-graph
	python setup.py install

Then, install HARP and the other requirements:

	git clone https://github.com/GTmac/HARP.git
	cd HARP
	pip install -r requirements.txt

# Usage
To run HARP on the *CiteSeer* dataset using *LINE* as the underlying network embedding model, run the following command:

``python src/harp.py --input example_graphs/citeseer/citeseer.mat --model line --output citeseer.npy --sfdp-path bin/sfdp_linux``

Parameters available:

**--input:** *input_filename*
1. ``--format mat`` for a Matlab .mat file containing an adjacency matrix.
By default, the variable name of the adjacency matrix is ``network``;
you can also specify it with ``--matfile-variable-name``.
2. ``--format adjlist`` for an adjacency list, e.g:

	``1 2 3 4 5 6 7 8 9 11 12 13 14 18 20 22 32``
	
	``2 1 3 4 8 14 18 20 22 31``
	
	``3 1 2 4 8 9 10 14 28 29 33``
	
	``...``

3. ``--format edgelist`` for an edge list, e.g:

	``1 2``
	
	``1 3``
	
	``1 4``
	
	``2 5``
	
	``...``

**--output:** *output_filename*
The output representations in Numpy ``.npy`` format.
Note that we assume the nodes in your input file are indexed **from 0 to N - 1**.

**--model** *model_name*
The underlying network embeddings model to use. Could be ``deepwalk``, ``line`` or ``node2vec``.
Note that ``node2vec`` uses the default parameters, which is p=1.0 and q=1.0.

**--sfdp-path** *sfdp_path*
Path to the binary file of SFDP, which is the module we used for graph coarsening.
You can set it to ``sfdp_linux``, ``sfdp_osx`` or ``sfdp_windows.exe`` depending on your operating system.

**More options:** The full list of command line options is available with ``python src/harp.py --help``.

# Evaluation
To evaluate the embeddings on a multi-label classification task, run the following command:

``python src/scoring.py -e citeseer.npy -i example_graphs/citeseer/citeseer.mat -t 1 2 3 4 5 6 7 8 9``

Where ``-e`` specifies the embeddings file, ``-i`` specifies the ``.mat`` file containing node labels,
and ``-t`` specifies the list of training example ratios to use.

# Note

SFDP is a library for multi-level graph drawing, which is a part of [GraphViz](http://www.graphviz.org).
We use SFDP for graph coarsening in this implementation.
Note that SFDP is included as a binary file under ``/bin``;
please choose the proper binary file according to your operation system.
Currently we have the binary files under OSX, Linux and Windows.

# Citation
If you find HARP useful in your research, please cite our paper:

	@inproceedings{harp,
		title={HARP: Hierarchical Representation Learning for Networks},
		author={Chen, Haochen and Perozzi, Bryan and Hu, Yifan and Skiena, Steven},
		booktitle={Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence},
		year={2018},
		organization={AAAI Press}
	}
## GraphGAN

- This repository is the implementation of [GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611) ([arXiv](https://arxiv.org/abs/1711.08267)):
> GraphGAN: Graph Representation Learning With Generative Adversarial Nets  
Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Xing Xie, Minyi Guo  
32nd AAAI Conference on Artificial Intelligence, 2018

![](https://github.com/hwwang55/GraphGAN/blob/master/framework.jpg)

GraphGAN unifies two schools of graph representation learning methodologies: generative methods and discriminative methods, via adversarial training in a minimax game.
The generator is guided by the signals from the discriminator and improves its generating performance, while the discriminator is pushed by the generator to better distinguish ground truth from generated samples.
	


### Files in the folder
- `data/`: training and test data
- `pre_train/`: pre-trained node embeddings
  > Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
- `results/`: evaluation results and the learned embeddings of the generator and the discriminator
- `src/`: source codes


### Requirements
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.8.0
- tqdm == 4.23.4 (for displaying the progress bar)
- numpy == 1.14.3
- sklearn == 0.19.1


### Input format
The input data should be an undirected graph in which node IDs start from *0* to *N-1* (*N* is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

##### txt file sample

```0	1```  
```3	2```  
```...```


### Basic usage
```mkdir cache```   
```cd src/GraphGAN```  
```python graph_gan.py```
