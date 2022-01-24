# Few-shot learning

This repository is built on top of [oscarknagg's code](https://github.com/oscarknagg/few-shot) and [rohitrango's code](https://github.com/rohitrango/STNAdversarial) on few shot learning. This is a pytorch implementation where we benchmark different few-shot learning methods three datasets, namely:
- Omniglot
- miniImageNet
- Fashion-products-dataset

See these Medium articles for some more information
1. [Theory and concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
2. [Discussion of implementation details](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)

# Contents
- [Setup](#setup)
   * [Requirements](#requirements)
   * [Training Data](#training-data)
   * [Train](#train)
   * [Results](#results)
      + [Prototypical Networks](#prototypical-networks)
      + [Matching Networks](#matching-networks)
      + [Model Agnostic Meta Learning](#model-agnostic-meta-learning)

- [Todo](#todo)

- [Citation](#citation)

- [Contact](#contact)



# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

### Training Data

1. Edit the `DATA_PATH` variable in `config.py` to the location where
you store the Omniglot, miniImageNet and fashion datasets.
2. Download the Omniglot dataset from https://github.com/brendenlake/omniglot/tree/master/python, place the extracted files into `DATA_PATH/Omniglot_Raw`
3. Download the miniImageNet dataset from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view, place in `DATA_PATH/miniImageNet/images`
4. Download the fashion dataset:\
**Note**: training parameters are currently tailored on fashion small. Training parameters for fashion large will be updated soon.

    * Download the fashion large dataset from https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1, place in `DATA_PATH/fashion-dataset` and rename `images` to `images_large`. 
    * Download the fashion small dataset from https://www.kaggle.com/paramaggarwal/fashion-product-images-small, place in `DATA_PATH/fashion-dataset` and rename `images` to `images_small`.

5. Run `scripts/prepare_omniglot.py` to prepare the Omniglot dataset.

6. Run `scripts/prepare_mini_imagenet.py` to prepare the miniImageNet dataset.

7. Run: 
    * `scripts/prepare_fashion.py --size large` to prepare the fashion small dataset.
    * `scripts/prepare_fashion.py --size small` to prepare the fashion large dataset.

5. After acquiring the data and running the setup scripts your folder structure should look
like
```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
    fashion/
        images_small/
        images_large/
        metaTest.txt/
        metaTrain.txt/
        styles.csv/
        images.csv/
```

### Train

To train all the models and reproduce the following results, run the following scripts:
```
sh proto_experiments.sh
sh matching_experiments.sh
sh maml_experiments.sh
```

# Results

### Prototypical Networks

The file `proto_experiments.sh` contains the hyperparameters 
used to obtain all the results given below.

![Prototypical Networks](https://github.com/oscarknagg/few-shot/blob/master/assets/proto_nets_diagram.png)


Run `experiments/proto_nets.py` to reproduce results from [Prototpyical
Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
(Snell et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet', 'fashion'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks
- augment: enables normal augmentation with configurable parameters in few_shots/dataset.py
- constrained: constrains STN space with parameters defined in few_shots/stn.py
- size: specific to the fashion. Chooses dataset variation.

To get experiments results, launch ```read_csv.py --file logs/**experiment_name** ```.

|                  | Omniglot |     |      |      |
|------------------|----------|-----|------|------|
| **k-way**        | **5**    |**5**|**20**|**20**|
| **n-shot**       | **1**    |**5**|**1** |**5** |
| Published        | 98.8     |99.7 |96.0  |98.9  |
| This Repo        | 98.2     |99.4 |95.8  |98.6  |

--k-train 60 --n-train 1

|                  | miniImageNet|     |
|------------------|-------------|-----|
| **k-way**        | **5**       |**5**|
| **n-shot**       | **1**       |**5**|
| Published        | 49.4        |68.2 |
| This Repo        | 48.0        |66.2 |

--k-train 20 --n-train 1

|                  | Fashion-SMALL  |     |      |      |      |       |
|------------------|----------------|-----|------|------|------|-------|
| **k-way**        | **2**          |**2**|**5** |**5** |**15**|**15** |
| **n-shot**       | **1**          |**5**|**1** |**5** |**1** |**5**  |
| This Repo        | 95.0           |99.0 |78.8  |91.6  |58.6  |76.9   |
| This Repo (aug)  | 95.0           |98.0 |79.0  |91.8  |55.6  |77.6   |

--k-train 20 --n-train 1         size=(80x80) epochs=200

|                  | Fashion-LARGE  |     |      |      |      |       |
|------------------|----------------|-----|------|------|------|-------|
| **k-way**        | **2**          |**2**|**5** |**5** |**15**|**15** |
| **n-shot**       | **1**          |**5**|**1** |**5** |**1** |**5**  |
| This Repo        | 93.5           |98.5 |78.8  |91.2  |58.8  |76.0   |
| This Repo (aug)  | 92.5           |97.1 |76.2  |88.2  |53.8  |71.9   |

--k-train 20 --n-train 1         size=(160,160) epochs 200

### Matching Networks

A differentiable nearest neighbours classifier.

![Matching Networks](https://github.com/oscarknagg/few-shot/blob/master/assets/matching_nets_diagram.png)

Run `experiments/matching_nets.py` to reproduce results from [Matching
Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
(Vinyals et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks
- fce: Whether (True) or not (False) to use full context embeddings (FCE)
- lstm-layers: Number of LSTM layers to use in the support set
    FCE
- unrolling-steps: Number of unrolling steps to use when calculating FCE
    of the query sample
- augment: enables normal augmentation with configurable parameters in few_shots/dataset.py
- constrained: constrains STN space with parameters defined in few_shots/stn.py
- size: specific to the fashion. Chooses dataset variation.

I had trouble reproducing the results of this paper using the cosine
distance metric as I found the converge to be slow and final performance
dependent on the random initialisation. However I was able to reproduce
(and slightly exceed) the results of this paper using the l2 distance
metric.

To get experiments results, launch ```read_csv.py --file logs/**experiment_name** ```.

|                     | Omniglot|     |      |      |
|---------------------|---------|-----|------|------|
| **k-way**           | **5**   |**5**|**20**|**20**|
| **n-shot**          | **1**   |**5**|**1** |**5** |
| Published (cosine)  | 98.1    |98.9 |93.8  |98.5  |
| This Repo (cosine)  | 92.0    |93.2 |75.6  |77.8  |
| This Repo (l2)      | 98.3    |99.8 |92.8  |97.8  |

--k-train 5 --n-train 1

|                        | miniImageNet|     |
|------------------------|-------------|-----|
| **k-way**              | **5**       |**5**|
| **n-shot**             | **1**       |**5**|
| Published (cosine, FCE)| 44.2        |57.0 |
| This Repo (cosine, FCE)| 42.8        |53.6 |
| This Repo (l2)         | 46.0        |58.4 |

--k-train 5 --n-train 1

|                  | Fashion-SMALL  |     |      |      |      |       |
|------------------|----------------|-----|------|------|------|-------|
| **k-way**        | **2**          |**2**|**5** |**5** |**15**|**15** |
| **n-shot**       | **1**          |**5**|**1** |**5** |**1** |**5**  |
| This Repo        | 95.0           |99.0 |79.0  |91.0  |59.1  |74.9   |

--k-train 40 --n-train 1 --fce False

### Model Agnostic Meta Learning (MAML)

![MAML](https://github.com/oscarknagg/few-shot/blob/master/assets/maml_diagram.png)

I used max pooling instead of strided convolutions in order to be
consistent with the other papers. The miniImageNet experiments using
2nd order MAML took me over a day to run.

Run `experiments/maml.py` to reproduce results from [Model-Agnostic
Meta-Learning](https://arxiv.org/pdf/1703.03400.pdf)
(Finn et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n: Support samples per class for few-shot tasks
- k: Number of classes in training tasks
- q: Query samples per class for training tasks
- inner-train-steps: Number of inner-loop updates to perform on training
    tasks
- inner-val-steps: Number of inner-loop updates to perform on validation
    tasks
- inner-lr: Learning rate to use for inner-loop updates
- meta-lr: Learning rate to use when updating the meta-learner weights
- meta-batch-size: Number of tasks per meta-batch
- order: Whether to use 1st or 2nd order MAML
- epochs: Number of training epochs
- epoch-len: Meta-batches per epoch
- eval-batches: Number of meta-batches to use when evaluating the model
    after each epoch
- augment: enables normal augmentation with configurable parameters in few_shots/dataset.py
- constrained: constrains STN space with parameters defined in few_shots/stn.py
- size: specific to the fashion. Chooses dataset variation.

NB: For MAML n, k and q are fixed between train and test. You may need
to adjust meta-batch-size to fit your GPU. 2nd order MAML uses a _lot_
more memory.

To get experiments results, launch ```read_csv.py --file logs/**experiment_name** ```.

|                  | Omniglot |     |      |      |
|------------------|----------|-----|------|------|
| **k-way**        | **5**    |**5**|**20**|**20**|
| **n-shot**       | **1**    |**5**|**1** |**5** |
| Published        | 98.7     |99.9 |95.8  |98.9  |
| This Repo (1)    | 95.5     |99.5 |92.2  |97.7  |
| This Repo (2)    | 98.1     |99.8 |91.6  |95.9  |

|                  | miniImageNet|     |
|------------------|-------------|-----|
| **k-way**        | **5**       |**5**|
| **n-shot**       | **1**       |**5**|
| Published        | 48.1        |63.2 |
| This Repo (1)    | 46.4        |63.3 |
| This Repo (2)    | 47.5        |64.7 |

|                  | Fashion-LARGE  |     |
|------------------|----------------|-----|
| **k-way**        | **2**          |**5**|
| **n-shot**       | **1**          |**1**|
| This Repo        | 76.2           |53.0 |

Number in brackets indicates 1st or 2nd order MAML.

### Todo

- [ ] Try different augmentation parameters
- [ ] Augment support set only and evaluate prototypical networks, matching networks and maml on fashion small and large
- [ ] Evaluate prototypical networks, matching networks and maml with well calibrated and unconstrained STNs
- [ ] Evaluate prototypical networks, matching networks and maml with manually constrained STNs.
- [ ] Evaluate prototypical networks, matching networks and maml with manually regularized and constrained STNs. 
- [ ] Repeat above experiments with fully conditional embeddings enabled matching networks
- [ ] Train second order MAML if resources are available. 


## Citation

Please consider to cite the following related papers if this repository helps you with your research:

```
@inproceedings{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in neural information processing systems},
  pages={4077--4087},
  year={2017}
}

@inproceedings{vinyals2016matching,
  title={Matching networks for one shot learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Wierstra, Daan and others},
  booktitle={Advances in neural information processing systems},
  pages={3630--3638},
  year={2016}
}

@article{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:1703.03400},
  year={2017}
}


@inproceedings{jena2020ma3,
  title={MA3: Model Agnostic Adversarial Augmentation for Few Shot learning},
  author={Jena, Rohit and Sukanta Halder, Shirsendu and Sycara, Katia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={908--909},
  year={2020}
}
```
## Contact

```
[Kamal Mustafa](kamal.mustafa.ks@gmail.com)
```
