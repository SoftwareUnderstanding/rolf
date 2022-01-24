# Performing Machine Translation using Apache MXNet
This repo uses Apache MXNet framework for performing Machine Translation.
Specifically, it tries to replace the usual softmax operator by hardmax in the attention mechanism for Decoder.

# Task
The Natural Language Processing task at hand is Machine Translation (as the title suggests).
It involves converting sequence of text from a source language into coherent and matching text in a target language.

# Model
This Machine Translation model use an Encoder-Decoder with Attention.

The Encoder-Decoder architecture with recurrent neural networks has become an effective and standard approach for both neural machine translation (NMT) and sequence-to-sequence (seq2seq) prediction in general.

Our model follows the common sequence-to-sequence learning framework with attention. It has three components: an encoder network, a decoder network, and an attention network.

— (Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation)[https://arxiv.org/abs/1609.08144], 2016

# Data

Standard datasets are required to develop, explore, and familiarize yourself with how to develop neural machine translation systems. The Europarl is a standard dataset used for statistical machine translation, and more recently, neural machine translation.

It is a collection of the proceedings of the European Parliament, dating back to 1996. Altogether, the corpus comprises of about 30 million words for each of the 11 official languages of the European Union

— (Europarl: A Parallel Corpus for Statistical Machine Translation, 2005)[http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf]

The creation of the dataset was lead by Philipp Koehn, author of the book (“Statistical Machine Translation")[http://amzn.to/2xbAuwx].

## Retrieving data

```
wget http://www.statmt.org/europarl/v7/fr-en.tgz
```
```
tar -xzvf fr-en.tgz
```

## Downsizing
Originally the files were 288MB long with 2007723 lines
```
du -h europarl-v7.fr-en.fr
288M	europarl-v7.fr-en.fr
```
```
$ wc -l europarl-v7.fr-en.fr
2007723 europarl-v7.fr-en.fr
```

For demonstration purpose, we shorten them.

```
head -100000 europarl-v7.fr-en.fr > tiny.europarl-v7.fr-en.fr
```

Result
tiny.europarl-v7.fr-en.fr and tiny.europarl-v7.fr-en.en
100000 lines and 16MB
