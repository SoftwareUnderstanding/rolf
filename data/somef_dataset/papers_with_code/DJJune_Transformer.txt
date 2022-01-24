# Transformer
This is the implement of `Transformer` which is proposed by 
`Attention is All You Need`(https://arxiv.org/abs/1706.03762)

## Requirements
  * Install python 3
  * Install pytorch == 0.4.0
  * Install tensorboardX
  * Install tqdm

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `module.py` contains all methods, including attention, feed-forward network and so on.
  * `network.py` contains networks including encoder, decoder.
  * `symbols.py` contains the definition of vocabulary.
  * `utils.py` contains the position encoding method.
  * `train.py` is a simple example for training a Transformer. 
  * `test.py` is a simple example for generating a sentence.
  * directory `checkpoints/` contains the models
  * directory `runs/` contains the training log file

## Training procedure visualization
I use `tensorborad` to show the loss curve and attention map
You can use the following commands to launch the service
```
tensorboard --logdir=./runs/*** --port=8888
```
Then open your browser and visit `localhost:8888`

## Training the network
  * Run `train.py`.


## Generate sentence
  * Run `test.py`.
  




