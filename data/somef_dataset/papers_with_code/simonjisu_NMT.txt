# Neural Machine Translation

Paper Implementation: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (v7 2016)

## Getting Started

### Prerequisites

```
pytorch 0.4.0
argparse 1.1
numpy 1.14.3
matplotlib 2.2.2
```

### Tutorial for NMT

* Jupyter notebook: [link](https://nbviewer.jupyter.org/github/simonjisu/NMT/blob/master/Neural_Machine_Translation_Tutorial.ipynb)
* Preparing for demo

### Result

Trained "IWSLT" 2016 dataset. Check [torchtext.dataset.IWSLT](https://torchtext.readthedocs.io/en/latest/datasets.html#iwslt) to download & loading dataset.

<p align="center">
  <img width="720" height="480" src="./pics/result.png">
</p>

### How to Start

For 'HELP' please insert argument behind `main.py -h`. or you can just run 

```
$ cd model
$ sh runtrain.sh
```

## Trainlog

I devide trains for 3 times, because of computing power.

After 1st training, load model and retrain at 2nd, 3rd time.

* Lowest losses & check points: 
* 1st train: [8/30] (train) loss 2.4335 (valid) loss 5.6971
* 2nd train: [1/30] (train) loss 2.3545 (valid) loss 5.6575
* 3rd train: [6/20] (train) loss 1.9401 (valid) loss 5.4970

you can see how i choose hyperparameters below

### Hyperparameters

| Hyperparameters |1st Train | 2st Train | 3st Train | Explaination | 
|--|--|--|--|--|
| BATCH| 50 | 50 | 50 | batch size | 
| MAX_LEN | 30 | 30 | 30 | max length of training sentences |
| MIN_FREQ | 2 | 2 | 2 | minimum frequence of words that appear in training sentences |
| EMBED | 256 | 256 | 256 | embedding size |
| HIDDEN | 512 | 512 | 512 | hidden size |
| ENC_N_LAYER | 3 | 3 | 3 | number of layer in encoder |
| DEC_N_LAYER | 1 | 1 | 1 | number of layer in decoder |
| L_NORM | True | True | True | whether to use layer normalization after embedding |
| DROP_RATE | 0.2 | 0.2 | 0.2 | dropout after embedding, if drop rate equal to 0, means not use it |
| METHOD | general | general | general | attention methods, "dot", "general" are ready to use |
| LAMBDA | 0.00001 | 0.00001 | 0.0001 | weight decay rate |
| LR | 0.001 | 0.0001 | 1.0 | learning rate |
| DECLR | 5.0 | 5.0 | - | decoder learning weight, multiplied to LR |
| OPTIM | adam | adam | adelta | optimizer algorithm |
| STEP | 30 | 20 | 20 | control learning rate at 1/3*step, 3/4*step by multiply 0.1 |
| TF | True | True | True | teacher forcing, whether to teach what token becomes next to model |

Please check train logs are in `trainlog` directory. 

## Todo:

* Layer Normalizaiton for GRU: https://discuss.pytorch.org/t/speed-up-for-layer-norm-lstm/5861
* seq2seq beam search: https://guillaumegenthial.github.io/sequence-to-sequence.html
* large output vocab problem: http://www.aclweb.org/anthology/P15-1001
* Recurrent Memory Networks(using Memory Block): https://arxiv.org/pdf/1601.01272
* BPE: https://arxiv.org/abs/1508.07909 

## License

This project is licensed under the MIT License 


