# summarisers

messing with some approaches to summarising text with machine learning

## pre-requisites

- Python 3.6.6 / pip 18.0 (I use Pyenv and virutalenv)
- Tensorflow

### Preparing the environment

The `setup.sh` script may be helpful in setting up your environment, assuming you have already installed `pyenv` and `virtualenv` (see my tutorial [Python dependency - hell no!](http://www.webpusher.ie/2018/09/19/python-dependency-hell-no/))

The script contains the following

```bash
pyenv install 3.6.6
pyenv rehash
pyenv virtualenv 3.6.6 summarisers
pyenv local summarisers

pip install -U pip

pip install -r requirements.txt

python -m spacy download en
```

Once you run that you should be ready to go.

## Datasets

You should download the datasets from https://www.kaggle.com/snapcrack/all-the-news

The scripts expect the files to be unzipped to the `datasets` folder in the root of the project.

There is a very basic data exploration script `dataexplore.py` that gives some basic insight into the data structure.

I've had to clean up some of the data

## Word embeddings

You will need to have downloaded the word embeddings. [On the GloVe official page](https://nlp.stanford.edu/projects/glove/) you should download the [GloVe pre-trained word embeddings](http://nlp.stanford.edu/data/glove.6B.zip). They are 822 Megs so it may take a while.

Unzip them to the datasets folder. You will see four files which contain the model weights for embeddings using four different vector dimensions.

```bash
glove.6B.50d.txt
glove.6B.100d.txt
glove.6B.200d.txt
glove.6B.300d.txt
```

## Example 1 - Sequence to Sequence

LSTM with Attention

I'll explain how to build a seq2seq model using LSTM

[You can read an overview in the example folder](./ex1/readme.md) and I have blogged / will blog about elements of this in more detail at [webpusher.ie](http://www.webpusher.ie)

### Building vocabulary vectors with word2vec

#### Compare output with sense2vec

## Example 2 - BiLSTM with Attention

## Example 3 - Seq2Seq with pointer and coverage

## other datasets

http://research.signalmedia.co/newsir16/signal-dataset.html

And also Newsroom https://summari.es/

## Longer summaries of larger source material

Requires a training set of summaries that are not a headline. There is a wikipedia dataset that may be useful

https://github.com/tscheepers/Wikipedia-Summary-Dataset

part of this https://github.com/tscheepers/CompVec

## Junk drawer

https://arxiv.org/pdf/1409.3215.pdf

https://scalableminds.com/blog/MachineLearning/2018/08/rasa-universal-sentence-encoder/

http://nlp.town/blog/anything2vec/

https://github.com/explosion/sense2vec

https://web.stanford.edu/class/cs224n/archive/WWW_1617/lecture_notes/cs224n-2017-notes6.pdf
