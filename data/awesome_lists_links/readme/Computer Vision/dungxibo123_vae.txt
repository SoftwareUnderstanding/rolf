# Auto-Encoding Variational Bayes

_Original Paper: [link](https://arxiv.org/abs/1312.6114)_


## Installation
* Recommend using an virtual environment to run
```bash
pip install -r requirements.txt
```

## Run

### Data set
Go to [Kaggle MNIST Dataset](https://www.kaggle.com/avnishnish/mnist-original) and download
Extract data file to get `mnist.mat`data file.

###### For Linux Shell

```shell
unzip archive.zip 
```


### Start to train the encoder and decoder
```shell
usage: python train.py [-h] -d DATA [-hd HIDDEN] [-ld LATENT] [-lr LEARNING] [-e EPOCHS] [-b BATCH_SIZE] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path/to/train/data
  -hd HIDDEN, --hidden HIDDEN
                        number of hidden unit
  -ld LATENT, --latent LATENT
                        number of latent unit
  -lr LEARNING, --learning LEARNING
                        learning rate
  -e EPOCHS, --epochs EPOCHS
                        epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -m MODEL, --model MODEL
                        path/to/model/saving/location
```

### After training
```shell
# Model class must be defined somewhere
model = torch.load("path/to/model/file/located")
model.eval()
```

