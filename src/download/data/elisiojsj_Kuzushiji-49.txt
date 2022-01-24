# Kuzushiji-49

## Kuzushiji-49 dataset

Kuzushiji are Japanese characters written in a cursive style, a script which is not taught anymore at school due to the modernization of the language.

The Kuzushiji dataset is created by the National Institute of Japanese Literature (NIJL), and is
curated by the Center for Open Data in the Humanities (CODH).

**Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. [arXiv:1812.01718](https://arxiv.org/abs/1812.01718)**

The set Kuzushiji-49, which I'm working on, contains 49 different classes of characters.

## STN

The model I'm using here is based on the Pytorch implementation of STN (Spatial Transformer Networks) which is a way to augment the dataset by allowing the neural network to enhance the geometric invariance of the model by performing spatial transformations on the input images.

**Deep Mind paper on STNs. [arXiv:1506.02025](https://arxiv.org/abs/1506.02025)**

## Software version
* conda 4.7.12
* python 3.7.4
* ptorch 1.3.1
