# End-To-End-Memory-Networks-in-Paddle

## 1. Introduction

This project reproduces [End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf)  based on paddlepaddle framework.

![模型简介](image/model_introduction.png)

Paper: [Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus: “End-To-End Memory Networks”, 2015.](https://arxiv.org/pdf/1503.08895v5.pdf)

Reference repo: [https://github.com/facebookarchive/MemNN](https://github.com/facebookarchive/MemNN)

The link of AiStudio: [https://aistudio.baidu.com/aistudio/projectdetail/2381004](https://aistudio.baidu.com/aistudio/projectdetail/2381004)

## 2. Results

The corresponding models are already included in this repo, under the directories `models_ptb` and `models_text8` respectively.

| Dataset | Paper Perplexity | Our Perplexity |
| :-----: | :--------------: | :------------: |
|   ptb   |       111        |     110.75     |
|  text8  |       147        |     145.62     |

## 3. DataSet

* Penn Treetank:

    * [Penn Treebank](https://aistudio.baidu.com/aistudio/datasetdetail/108805) 

        train：887k words

        valid：70k words

        test：78k words

        vocabulary  size：10k

    * [text8](https://aistudio.baidu.com/aistudio/datasetdetail/108807)

        train：A total of 100M characters are divided into 93.3M/5.7M /1M characters for train/valid/test. Replace words that occur less than 10 times with <UNK>.

## 4. Environment

* Hardware: GPU
* Framework: Paddle >= 2.0.0, progress

## 5. Quick Start

### train

The training parameters can be adjusted in the `config.py` file.

Note: Since this model is greatly affected by random factors, the results of each training are quite different. Even if random seeds are fixed, the training results cannot be completely consistent due to GPU.

#### train on ptb dataset

```bash
cp config/config_ptb config.py
python train.py
```

#### select the best model

Since the model is greatly affected by random factors, many times of training are needed to find the optimal model. In the original paper, 10 times of training are conducted on the ptb dataset, and the model with the best performance on the test set is retained. This replay provides a script to train multiple times to get a model with sufficient accuracy.

The following is the [log](./log/ptb_train_until.log) of multiple trainings on the ptb dataset to achieve the target accuracy.

#### train on text8 dataset

```bash
cp config/config_text8 config.py
python train.py
```

### eval

Keep the `config.py` file as it was during training

```
python eval.py
```

### Prediction using pre training model

#### on ptb dataset

```bash
cp config/config_ptb_test config.py
python eval.py
```

results:

![](image/test_ptb.png)

#### on text8 dataset

```bash
cp config/config_text8_test config.py
python eval.py
```

results:

![](image/test_text8.png)

## 6. Code structure

### 6.1 structure

```
├── checkpoints
├── config
│   ├── config_ptb
│   ├── config_ptb_test
│   ├── config_text8
│   └── config_text8_test
├── data
│   ├── ptb.test.txt
│   ├── ptb.train.txt
│   ├── ptb.valid.txt
│   ├── ptb.vocab.txt
│   ├── text8.test.txt
│   ├── text8.train.txt
│   ├── text8.valid.txt
│   └── text8.vocab.txt
├── models_ptb
│   └── model_17814_110.75
├── models_text8
│   └── model_500_7_100_145.62
├── image
│   ├── model_introduction.png
│   ├── test_ptb.png
│   └── test_text8.png
├── log
│   └── ptb_train_until.log
├── README_cn.md
├── README.md
├── requirements.txt
├── config.py
├── model.py
├── data.py
├── train.py
├── eval.py
├── train_until.py
└── utils.py
```

### 6.2 Parameter description

You can set the following parameters in `config.py`

```
config.edim = 150                       # internal state dimension
config.lindim = 75                      # linear part of the state
config.nhop = 7                         # number of hops
config.mem_size = 200                   # memory size
config.batch_size = 128                 # batch size to use during training
config.nepoch = 100                     # number of epoch to use during training
config.init_lr = 0.01                   # initial learning rate
config.init_hid = 0.1                   # initial internal state value
config.init_std = 0.05                  # weight initialization std
config.max_grad_norm = 50               # clip gradients to this norm
config.data_dir = "data"                # data directory
config.checkpoint_dir = "checkpoints"   # checkpoint directory
config.model_name = "model"             # model name for test and recover train
config.recover_train = False            # if True, load model [model_name] before train
config.data_name = "ptb"                # data set name
config.show = True                      # print progress, need progress module
config.srand = 17814                    # initial random seed
```

