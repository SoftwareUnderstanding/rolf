# Machine translation with PyTorch
<p align="center">
  <img src="https://github.com/Andrey885/Machine_translation_PyTorch/blob/master/gotta_torch.png" alt="drawing" width="300"/>
</p>

[![Build Status](https://github.com/Andrey885/Machine_translation_PyTorch/blob/master/.github/workflows/build-passing.svg)](https://github.com/Andrey885/Machine_translation_PyTorch/actions)

## Introduction
The attention mechanism is a popular easy-to-implement model architecture
designed to perform well on different NLP tasks, including machine translation.
 Empowered with pretrained [SpaCy](#https://spacy.io/) word model this
approach makes a strong baseline with comparable to state-of-the-art perfomance.
In this repo the translator from German to English is trained and demonstrated. In case
you need other languages support, thanks to SpaCy team there are
 plenty of available language models to train the exact same model on.
 Pretrained German-English model is already available after setup.


 ## Major dependencies
 1. Python 3.8

 2. PyTorch 1.5.0

 3. SpaCy 2.2.4

## Contents
1. [Setup](#Setup)
2. [Dataset](#Dataset)
3. [Train model](#Train-model)
4. [Hyperparameter tuning](#Hyperparameter-tuning)
5. [Results&Demo](#Results&Demo)
6. [Source](#Source)


## Setup

Use automatic setup script:
```
bash setup.sh
```

(If encounter a problem with Windows line endings, run *sed -i 's/\r//g' setup.sh* first)

or complete the installation manually in four steps (which is also easy):

 1. Install the required python packages:

```
python -m pip install -r requirements.txt
```

 2. Download and install pretrained Spacy language models:
```
sudo python -m spacy download en
sudo python -m spacy download de_core_news_sm
```

 3. Clone submodules

 ```
 git submodule init
 git submodule update
 ```

 4. Download a [pretrained model](https://drive.google.com/uc?id=1rNYfjFcSnp3Mi5sv0CL4Q9w6lQVlxhMh)
  and please kindly put it in *checkpoints* folder.

## Dataset

Use [Multi30k](https://github.com/multi30k/dataset) translation dataset available from [PyTorch](https://torchtext.readthedocs.io/en/latest/datasets.html) - a
 small dataset from 2016 year challenge. The training is done on de-en part of it. The dataset statistics is the following:

```
train:
 (en) 29000 sentences, 377534 words, 13.0 words/sent
 (de) 29000 sentences, 360706 words, 12.4 words/sent
 val:
 (en) 1014 sentences, 13308 words, 13.1 words/sent
 (de) 1014 sentences, 12828 words, 12.7 words/sent
 test:
 (en) 1000 sentences, 11376 words, 11.4 words/sent
 (de) 1000 sentences, 10758 words, 10.8 words/sent
```

Example pair of sentences:
```
 ein mädchen in einem karateanzug bricht ein brett mit einem tritt .
 a girl in karate uniform breaking a stick with a front kick .
```

## Train model

You can simply start training the model with this terminal command:
```
python train.py
```
Default arguments are set to optimal: see
 [Hyperparameter tuning](#Hyperparameter-tuning) section.
However, you are encouraged to make your own experiments.

This script will be saving models in ```./checkpoints/``` and writing logs in ```./logs/``` folders.
 Best pretrained model is already available after setup at ```./checkpoints/en_de_final.pt```

## Hyperparameter tuning

Several experiments on model hyperparameters were held.
The training curves may be found on
[tensorboard dev](https://tensorboard.dev/experiment/ksbaLHxzRgqGgPlbE5kWqw/).

We acquired the following table:

| Experiment id | hidden_size | pf_dim | n_heads | n_layers | Bleu score
|---|---|---|---|---|---|
| 1 | 256 | 512 | 8 | 3 | 0.3390
| 2 | 128 | 512 | 8 | 3 | 0.3507
| 3 | 64 | 512 | 8 | 3 | 0.3353
| 4 | 128 | 1024 | 8 | 3 | **0.3582**
| 5 | 256 | 2048 | 8 | 3 | 0.3385
| 6 | 128 | 1024 | 4 | 3 | 0.3557
| 7 | 128 | 1024 | 16 | 3 | 0.3464
| 8 | 128 | 1024 | 8 | 4 | 0.3494
| 9 | 128 | 1024 | 8 | 2 | 0.3460

## Results&Demo

The model is capable of producing decent results on samples from test set,
 achieving **0.3582** Bleu score on val and **0.3347** on test 
  sets (with experiment id 4 config) of de-en part of Multi30k dataset, which indicates a nice level of perfomance.

Run and see how it works:

```
python demo.py
```

Some sample results:

```
Input: eine straße neben einem interessanten ort mit vielen säulen .
GT translation: a road next to an interesting place with lots of pillars .
Model output: a street next to a plaza with many interesting pillars .

Input: ein skateboarder in einem schwarzen t-shirt und jeans fährt durch die stadt .
GT translation:  a skateboarder in a black t - shirt and jeans skating threw the city .
Model output:  a skateboarder in a black t - shirt and jeans is riding through the city .
```

## Source

Tutorial with awesome model architectures:

https://github.com/bentrevett/pytorch-seq2seq

Paper with the description of attention model:

https://arxiv.org/abs/1706.03762

**Also useful tutorials**:

Nice short book to understand NLP basics (awful for production and demo, however):

https://github.com/joosthub/PyTorchNLPBook

Tutorials from good PyTorch folks, also nice and simple to get started:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
