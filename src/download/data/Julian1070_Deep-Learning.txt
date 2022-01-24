# Deep Learning
This repository holds some deep learning projects I have worked on in class.

## Neural Network from scratch with numpy
I built a neural network from scratch using only numpy and trained it on the dataset from the famous Titanic challenge on Kaggle. Here is the submission:

![Screenshot of Kaggle submission](https://github.com/Julian1070/Deep-Learning/blob/master/TitanicKaggle/np_nn_submission.png "Screenshot of Kaggle submission")

## GAN generating flag icons
With the help of diegoalejogm's implementation of a Vanilla GAN generating hand-written digits (https://github.com/diegoalejogm/gans/blob/master/1.%20Vanilla%20GAN%20PyTorch.ipynb), I used a dataset of small flag icons from flagpedia (http://flagpedia.net/download) to generate small black and white flag icons.

**Examples of the training data:**

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/ad.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/ad.png" width="160" height="100" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/ae.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/ae.png" width="200" height="100" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/af.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/flags/af.png" width="150" height="100" />

**Examples of the resulting images (after 8000 epochs):**

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag1.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag1.png" width="255" height="100" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag3.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag3.png" width="255" height="100" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag6.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/FlagGAN/generated_flags/generated_flag6.png" width="255" height="100" />

## Comment classification with LSTM and GRU
This Pytorch LSTM and GRU perform multi-class categorization on online comments from a kaggle challenge (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The comments are categorized in the following **six categories**:
- toxic
- severe toxic
- obscene
- threat
- identity hate

These models were built after the end of the challenge, so the results are based on the actual test set data, which was made public after the deadline.

#### LSTM

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/LSTMLabelPerformance.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/LSTMLabelPerformance.png" width="400" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/LSTMCommentPerformance.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/LSTMCommentPerformance.png" width="400" />

#### GRU

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/GRULabelPerformance.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/GRULabelPerformance.png" width="400" />

<img src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/GRUCommentPerformance.png" data-canonical-src="https://github.com/Julian1070/Deep-Learning/blob/master/NLPClassification/results/GRUCommentPerformance.png" width="400" />

## ULMFiT
In this notebook, I am applying the pretrained model from Jeremy Howard & Sebastian Ruder's paper "Universal Language Model Fine-tuning for Text Classification" (2018, see https://arxiv.org/abs/1801.06146) to the twitter airline sentiment dataset from Kaggle (https://www.kaggle.com/crowdflower/twitter-airline-sentiment/downloads/twitter-airline-sentiment.zip/2). In my implementation I am borrowing from fastai's tutorial on ULMFiT (https://docs.fast.ai/text.html).
