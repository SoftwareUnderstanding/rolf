# Differentiable Sampling with Flexible Reference Word Order for Neural Machine Translation
This is the code we used in our paper:

[Differentiable Sampling with Flexible Reference Word Order for Neural Machine Translation](https://www.aclweb.org/anthology/N19-1207/)

Weijia Xu, Xing Niu, Marine Carpuat

## Requirements
[![PyPI version](https://badge.fury.io/py/sockeye.svg)](https://badge.fury.io/py/sockeye)
[![GitHub license](https://img.shields.io/github/license/awslabs/sockeye.svg)](https://github.com/awslabs/sockeye/blob/master/LICENSE)

This is implemented based on [Sockeye](https://awslabs.github.io/sockeye/).

## Data Processing
[IWSLT'14 German-English data](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) preprocessed using the [script](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh) by Ranzato et al. (2015).

[IWSLT'15 Vietnamese-English data](https://nlp.stanford.edu/projects/nmt) preprocessed by Luong and Manning (2015).

## Running
For information on training and inference with Sockeye, please visit [the documentation](https://awslabs.github.io/sockeye/).
