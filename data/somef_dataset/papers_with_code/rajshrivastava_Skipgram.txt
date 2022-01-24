# Skipgram

## Table of Contents
  * [Overview](#overview)
  * [How to run](#how-to-run)
  * [How does it work?](#how-does-it-work-)

## Overview
Word2Vec recognizes semantic closeness between words by transforming words into vectors with meaningful contextual information.

Mikolv et. al. in https://arxiv.org/abs/1301.3781 proposed two architectures for Word2Vec:
- Skip-gram
- CBOW

<img src="./res/imgs/models.png" alt="models" width="400"/>


The original code was written in C.

This repository contains the implementation of neural network for skip-gram from scratch in Python without using any machine learning or text processing libraries.

<img src="./res/imgs/skipgram.png" alt="skipgram" width="400"/>

## How to run
  To train the model, run the `train_minibatch.py` script on command line:
  ```
  python train_minibatch.py
  ```

  To predict similar words, run the `predict.py` script on command line:
  ```
  python predict.py
  ```

## How does it work?
1. "train_minibatch.py" is the training file. It trains the neural network for any given dataset (`dataset.csv`) and generates `skipgram_w1.npy`, `initialPlot.png` (word embeddings of untrained word vectors), and `finalPlot.png` (word embeddings for trained word vectors).

2. The resultant trained word vectors  are preserved as `skipgram_w1.npy`.

3. `predict.py` uses the trained word vectors to:
    - output cosine similarity between two input words.
    - output 10 closest context words to any input words. This code has been formatted to fetch input from command line.

After this, I implemented [another version using TensorFlow](https://github.com/rajshrivastava/Word-embeddings).
