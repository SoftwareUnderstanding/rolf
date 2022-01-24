# Abstractive summarization with OpenNMT-py

This is a [Pytorch](https://github.com/pytorch/pytorch)
implementation of Abstractive summarization methods on top
of [OpenNMT](https://github.com/OpenNMT/OpenNMT). It features vanilla attention seq-to-seq LSTMs,
[pointer-generator networks (See 2017)](https://arxiv.org/abs/1704.04368) ("copy attention"),
as well as [transformer networks  (Vaswani 2017)](https://arxiv.org/pdf/1706.03762.pdf)  ("attention is all you need")
as well as instructions to run the networks on both the Gigaword and the CNN/Dayly Mail datasets.


Table of Contents
=================

  * [Requirements](#requirements)
  * [Implemented models](#implemented-models)
  * [Quickstart](#quickstart)
  * [Results](#results)
  * [Pretrained models](#pretrained-models)

## Requirements

```bash
pip install -r requirements.txt
```

## Implemented models

The following models are implemented:

- Vanilla attention LSTM encoder-decoder
- Pointer-generator networks: ["Get To The Point: Summarization with Pointer-Generator Networks",
  See et al., 2017](http://arxiv.org/abs/1704.04368)
- Transformer networks: ["Attention is all you need", Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762)

## Quickstart

### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -share_vocab -dynamic_dict -src_vocab_size 50000
```

The data can be either Gigaword or the CNN/Daily Mail dataset. For CNN/daily mail, it is also recommended to truncate inputs and outputs: -src_seq_length_trunc 400 -tgt_seq_length_trunc 100

The data consists of parallel source (`src`) and target (`tgt`) data containing one example per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

For Gigaword, download the data from : https://github.com/harvardnlp/sent-summary. Then, extract it (```tar -xzf summary.tar.gz ```)
For CNN/Daily Mail, we assume access to such files. Otherwise, these can be built from https://github.com/OpenNMT/cnn-dailymail.

Validation files are required and used to evaluate the convergence of the training.

After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

The basic command would be:

```bash
python train.py -data data/demo -save_model demo_model -share_embeddings
```

The main relevant parameters to be changed for summarization are:

* pointer\_gen to enable Pointer Generator
* -encoder_type transformer -decoder_type transformer to enable Transformer networks
* word\_vec\_size (128 has given good results)
* rnn\_size (256 or 512 work well in practice)
* encoder\_type (brnn works best on most models)
* layers (1 or 2, up to 6 on transformer)
* gpuid (0 for the first gpu, -1 if on cpu)

The parameters for our trained models are described below

### Step 3: Summarize

```bash
python translate.py -model demo-model_epochX_PPL.pt -src data/src-test.txt -o output_pred.txt -beam_size 10
-dynamic_dict -share_vocab
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

### Step 4: Evaluate with ROUGE

Perplexity and accuracy are not the main evaluation metrics for summarization. Rather, the field uses
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))
To evaluate for rouge, we use [files2rouge](https://github.com/pltrdy/files2rouge), which itself uses
[pythonrouge](https://github.com/tagucci/pythonrouge).

Installation instructions:

```bash
pip install git+https://github.com/tagucci/pythonrouge.git
git clone https://github.com/pltrdy/files2rouge.git
cd files2rouge
python setup_rouge.py
python setup.py install
```

To run evaluation, simply run:
```bash
files2rouge summaries.txt references.txt
```
In the case of CNN, evaluation should be done with beginning and end of sentences tokens stripped.


## Results

#### Gigaword:
|   | Rouge-1  | Rouge-2   | Rouge-L   |
|---|---|---|---|
| Attention LSTM | **35.59** | **17.63** | **33.46** |
| Pointer-Generator | 33.44  | 16.55  | 31.43 |
| Transformer | 35.10  | 17.01  | 33.09 |
| lead-8w baseline  | 21.31  | 7.34 | 19.95  |

Detailed settings:

|   | Attention | Pointer-Generator | Transformer |
|---|---|---|---|
| Vocabulary size| 50k | 50k | 50k |
| Word embedding size | 128  | 128 | 512 |
| Attention | MLP  | Copy  | Multi-head |
| Encoder layers | 1 | 1 | 4 |
| Decoder layers | 1 | 1 | 4 |
| Enc/Dec type | BiLSTM | BiLSTM | Transformer |
| Enc units | 512 | 256 | 512 |
| Optimizer | Sgd | Sgd | Adam |
| Learning rate | 1 | 1 | 1 |
| Dropout | 0.3 | 0.3 | 0.2 |
| Max grad norm | 2 | 2 | n.a |
| Batch size | 64 | 32 | 32 |

The Transformer network also has the following additional settings during training:
> `param_init=0 position_encoding warmup_steps=4000 decay_method=noam`

#### CNN/Daily Mail

|   | Rouge-1  | Rouge-2   | Rouge-L   |
|---|---|---|---|
| Attention LSTM | 30.25 | 12.41 | 22.93 |
| Pointer-Generator | **34.00**  | **14.70**  | **36.57** |
| Transformer | 23.90  | 5.85  | 17.36 |
| lead-3 baseline  | 40.34 | 17.70 | 36.57  |

Detailed settings:

|   | Attention | Pointer-Generator | Transformer |
|---|---|---|---|
| Vocabulary size| 50k | 50k | 50k |
| Word embedding size | 128  | 128 | 256 |
| Attention | MLP  | Copy  | Multi-head |
| Encoder layers | 1 | 1 | 4 |
| Decoder layers | 1 | 1 | 4 |
| Enc/Dec type | BiLSTM | BiLSTM | Transformer |
| Enc units | 256 | 256 | 256 |
| Optimizer | Sgd | Sgd | Adam |
| Learning rate | 1 | 1 | 1 |
| Dropout | 0.3 | 0.3 | 0.2 |
| Max grad norm | 2 | 2 | n.a |
| Batch size | 32 | 32 | 64 |

The Transformer network also has the following additional settings during training:
> `param_init=0 position_encoding warmup_steps=4000 decay_method=noam`


## Pretrained models

### Gigaword

* [Attention LSTM](https://drive.google.com/file/d/1PFrcrn_9HN-Ww0nFaZL3WDzIyhZ73P6w/view?usp=sharing)
* [Pointer-Generator](https://drive.google.com/file/d/1Wlmnpdx7dmoG49YpCfODtAt5fgm7K_dg/view?usp=sharing)
* [Transformer](https://drive.google.com/file/d/1NHobKlg3JlPpltg9KJfcPliwOtCgiwRc/view?usp=sharing)

### CNN/Daily Mail

* [Attention LSTM](https://drive.google.com/file/d/1PHWH20LsoKyUiPq4xnWu7SF1o72Frouu/view?usp=sharing)
* [Pointer-Generator](https://drive.google.com/file/d/1pS5G9_Usb_GueKMQYyAUH0j0Pk_hAQHZ/view?usp=sharing)
* [Transformer](https://drive.google.com/file/d/18C6RwEm87KSNkhvu58caF41-Kg-dtAMJ/view?usp=sharing)
