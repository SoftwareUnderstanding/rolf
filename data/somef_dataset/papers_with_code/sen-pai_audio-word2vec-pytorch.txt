# Reproducing Audio-Word2Vec
Sequence-to-sequence neural network.
Try out the ToyDataset to understand how it works.
Feed MFCC's instead to train Audio-Word2Vec.


Adapted from https://github.com/b-etienne/Seq2seq-PyTorch/
Check it out if you are looking for a good repo on Seq2Seq
## Original papers

* https://arxiv.org/abs/1409.0473
* https://arxiv.org/abs/1508.04025

## Getting Started


### Prerequisites

Install the packages with pip

```
pip install -r requirements.txt
```

### Train model

Train and evaluate models with

```
python main.py --config=<json_config_file>
```
Examples of config files are given in the "experiments" folder. All config files have to be placed in this directory.
### Hyper-parameters

You can tune the following parameters:

* decoder type (with or without Attention)
* encoder type (with or without downsampling, with or without preprocessing layers)
* the encoder's hidden dimension
* the number of recurrent layers in the encoder
* the encoder dropout
* the bidirectionality of the encoder
* the decoder's hidden dimension
* the number of recurrent layers in the decoder
* the decoder dropout
* the bidirectionality of the decoder
