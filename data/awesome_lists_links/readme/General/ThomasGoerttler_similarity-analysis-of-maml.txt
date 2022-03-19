 # Exploring the Similarity of Representations in Model-Agnostic Meta-Learning
 
 This repository contains the code of the paper [Exploring the Similarity of Representations in Model-Agnostic Meta-Learning (Goerttler and Obermayer, Learning to Learn at ICLR 2021)](https://openreview.net/forum?id=yOQbCLSWg0b). 
 
 It includes the code to both train and analyze the models which are anaylzed in the paper.
 The repository is based on the [code](https://github.com/cbfinn/maml) of original paper on MAML [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400). It includes code for running the few-shot supervised learning domain experiments, including sinusoid regression, Omniglot classification, and MiniImagenet classification.

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in ```data/omniglot_resized/resize_images.py``` and ```data/miniImagenet/proc_images.py``` respectively.

### Usage

The models we trained are available at [code](https://tubcloud.tu-berlin.de/apps/files/?dir=/&fileid=749323232772303)
**Note:** Unfortunately the services of TU Berlin are currently down due too an [attack on the system](https://www.campusmanagement.tu-berlin.de/zecm/). Therefore, the weights are not accesable currently.

You can train your own model with our setup as well

Omniglot example in the paper: 

```python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=./logs/MAML/omniglot20way/```

MiniImageNet example in the paper: 

```python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=./logs/MAML/miniimagenet1shot/ --num_filters=32 --max_pool=True```


To reproduce the analysis done in the paper use (after training the model)
for Figure 1:

```python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=/mnt/data/ni/goerttler/logs/MAML/omniglot20way/ --train=False --analyze=True --points_to_analyze 50 --base_analysis --steps_to_analyze range(0,60000,1000)```

for Figure 2

```python mnist.py```

For Figure 3 and 5

```python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=/mnt/data/ni/goerttler/logs/MAML/omniglot20way/ --train=False --analyze=True --points_to_analyze 50 --base_analysis --steps_to_analyze range(0,60000,1000)```

For Figure 4

```python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=/mnt/data/ni/goerttler/logs/MAML/omniglot20way/ --train=False --analyze=False```

For Figure 6

```python main.p --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=/mnt/data/ni/goerttler/logs/MAML/miniimagenet1shot/ --num_filters=32 --max_pool=True -train=False --analyze=True --steps_to_analyze range(20000,70000,10000)```

