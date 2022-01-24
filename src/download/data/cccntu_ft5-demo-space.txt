---
title: FT5 News Summarizer
emoji: ✍️
colorFrom: yellow
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# FT5 News Summarizer

This is a proof of concept for FT5, a novel hybrid model based on FNet and T5. The idea is first proposed [on the huggingface forum](https://discuss.huggingface.co/t/train-hybrid-fnet-for-generation-fnet-mixing-tokens-with-fourier-transforms/7622).

## Introduction

FNet (https://arxiv.org/abs/2105.03824) is an attention-free model that is faster to train, and uses less memory. It does so by replaces self-attention with unparameterized Fourier Transform, but custom implementation is required to do causal (unidirectional) mask. In this project, we instead explore a hybrid approach. We use a encoder-decoder architecture, where the encoder is an attention-free, using Fourier Transform, and the decoder uses cross-attention and decoder self-attention.

## Model
The model architecture is based on T5 (https://arxiv.org/abs/1910.10683), except the encoder self-attention is replaced by fourier transform as in FNet.
Our implementation can be found here:
* https://github.com/cccntu/transformers/compare/7736fad96135a7eff55932ce974b7be2a74fcb1d..54d12d5e88e5bc0da75f35c109a4424b1afcdf66

## Pre-Training

* We use the T5 unsupervised objective, and train for the same number of steps, with the same sequence length.
* Data: [OpenWebText (huggingface link)](https://huggingface.co/datasets/openwebtext)
  * I chose this dataset because it's sufficiently large (pre-training iterates over the dataset ~4 times), but not too large (makes coding easier, so we could start training ealier), it's clean, and it's directly loadable on huggingface.
* Training code: https://github.com/cccntu/fnet-generation/blob/main/scripts/run_t5_mlm_flax.py
  * Note: This is >2x faster than the original script on TPU. I found the preprocessing is the bottleneck because TPU is so fast, so I used PyTorch Dataloader to parallelize it. I also refactored training loop.

## Fine-Tuning

* It is fine-tuned on [CNN Dailymail Dataset (huggingface link)](https://huggingface.co/datasets/cnn_dailymail). I chose this data because it's used in original T5 paper, so it can be directly compared.
* The fine-tuning hyperparameter again follows T5, except it is manually ealry-stopped, since CNN/DM is a much smaller dataset and model overfits at a fraction of training time. The best checkpoint is selected by `rouge2` on validation set.

## Experiments

### T5-base

A unmodified T5-base is first trained to establish a baseline and make sure the training code is correct.
The pre-trained model is here: [flax-community/t5-base-openwebtext](https://huggingface.co/flax-community/t5-base-openwebtext/)
And the model fine-tuned on CNN/DM is here: [flax-community/t5-base-cnn-dm](https://huggingface.co/flax-community/t5-base-cnn-dm)

Pre-training takes 59 hours on TPUv3x8, with the same batch size, training steps, and optimizer as in T5 paper.
In T5 paper, fine-tuning uses half of the training step as pre-training, but the model already starts overfitting at 5 hours.

### FT5-base

The pre-trained model is here: [flax-community/ft5-base-openwebtext](https://huggingface.co/flax-community/ft5-base-openwebtext/)
And the model fine-tuned on CNN/DM is here: [flax-community/ft5-cnn-dm](https://huggingface.co/flax-community/ft5-cnn-dm)


### Results

Our best (decided by vaidation rouge-2) checkpoints achieves rouge-2 of 18.61 (t5) and 16.5 (ft5) on the test set of CNN/DM.
It is lower than the numbers reported by T5 paper. We found 2 major difference that might be the cause.
1. During pre-training, I mistakenly used 1e-3 as learning rate, whereas t5 uses 1e-2 as learning rate. Later I did a another partial run with learning rate = 1e-2 and found it converges to a lower pre-traing loss.
2. During evaluation, T5 paper uses `beam width of 4 and a length penalty of α = 0.6` to generate output, and we used greedy decoding.


## Discussion & Future work

Our resulting model trains ~1.3x faster (steps/seconds) on TPU. To follow the original T5 settings, we used sequence length = 512, but FNet suggests Fourier Transform is even faster (relative to self-attention) at longer sequence length, and more efficient (relative) on GPU. I think it is a promising direction to apply FT5 to tasks that require longer inputs.

At inference time (CPU, non-batched), we found FT5 starts generating tokens with less latency (time to generate the first token), but when generating longer text, the speed difference is less significant. And with beam search or reranking, we need to generate the full text before outputting. This suggests FT5 is more suitable for tasks that can be done in batch on GPU.

