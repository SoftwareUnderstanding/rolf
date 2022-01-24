# BERT-on-Pytorch-Template

Pytorch implementation of Google AI's 2018 BERT on [moemen95's Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template).

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## Overview

### Pytorh Template

![](https://github.com/moemen95/Pytorch-Project-Template/raw/master/utils/assets/class_diagram.png)

[moemen95's Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template) has a specific structure represented above. It's proposing a baseline for any Pytorch project so that we can only focus on the model implementation. It provides some examples as well. So click the [link](https://github.com/moemen95/Pytorch-Project-Template) and see what it is.

### BERT

This repository is a **reconstruction result** of [dhlee347's Pytorchic BERT](https://github.com/dhlee347/pytorchic-bert) and [codertimo's BERT-pytorch](https://github.com/codertimo/BERT-pytorch) on **Pytorch template**. The purpose of this is to learn how pytorch and bert work. So in this repository, `pretraining` and `validating` are only available. 

To understand BERT, I recommend to read articles below.

> (English)
> * [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
> * [The illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
> * [Intuitive Understanding of Attention Mechanism in Deep Learning](https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f)
> * [Attention and memory in deep learning and nlp](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

> (Korean)
> * [BERT 톺아보기](http://docs.likejazz.com/bert/#input-embeddings)
> * [The illustrated transformer (translated)](https://nlpinkorean.github.io/illustrated-transformer/)
> * [Attenstion Mechanism and transformer](https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225)


### Tasks

In the paper, authors use `masked language model` and `predict next sentence` tasks for pretraining. Here's short explanation of those two (copied from [codertimo's BERT-Pytorch](https://github.com/codertimo/BERT-pytorch)).

#### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

#### Predict Next Sentence

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = Not Next
```

"Is this sentence can be continuously connected?"

understanding the relationship, between two text sentences, which is not directly captured by language modeling


## Results

```
Iter (loss=8.964 / NSP_acc=0.302): 100%|███████████████████████████████████████████████| 2746/2746 [36:34<00:00,  1.37it/s]
[INFO]: Epoch 1/50 : Average Loss 16.002 / NSP acc: 0.506
Iter (loss=4.536 / NSP_acc=0.281): 100%|███████████████████████████████████████████████| 2746/2746 [36:28<00:00,  1.37it/s]
[INFO]: Epoch 2/50 : Average Loss 7.178 / NSP acc: 0.526
Iter (loss=3.408 / NSP_acc=0.260): 100%|███████████████████████████████████████████████| 2746/2746 [36:31<00:00,  1.29it/s]
[INFO]: Epoch 3/50 : Average Loss 4.440 / NSP acc: 0.544
```

In pretraining with Korean corpus(sejong corpus), 300k iteration with 32 batch size, I was able to get 78% of accuracy in Next Sentence Prediction task.
The average loss goes down to 2.643.

With Korean corpus, the result of using 32 batch size is better than using 96. It seems that more frequent parameter updating leads to the optima. Pictures below are loss graphs of `Language Model Loss` and `Next Sentence Prediction Classification Loss`. 

![loss_lm](https://user-images.githubusercontent.com/12293076/63478132-6b565e80-c4c3-11e9-9c58-2830f111b05e.png)

![accuracy](https://user-images.githubusercontent.com/12293076/63477724-6db7b900-c4c1-11e9-841c-52c4bc8b7408.png)

The result shows that the model had been learning about NSP task after language model because of the difference of magnitude of loss values.

I'm preparing English corpus for another experiment. 

## Usage

### Prepare your corpus

Basically, your corpus should be prepared with two sentences in one line with tab(\t) separator
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```

### Edit configs

In `configs/bert_exp_0.json`, you can edit almost all hyper-parameters.

### Making vocab

If you are fine to use `Byte Pair Encoding`, it will generate vocab file according to your corpus. If else, you need to build your own. While the model runs, it will do basic text cleaning and tokenization of the corpus by `BPE`. You will find the `model` and `vocab` file of `BPE` in `experiment/bert_exp_0` directory.

### run

Run `run.sh`.


