transformers_from_scratch
==============================

Simple implementation of transformers modules for learning its structure and for personal practicing purposes.

Based on:
* https://arxiv.org/abs/1706.03762
* https://www.youtube.com/watch?v=iDulhoQ2pro 
* http://jalammar.github.io/illustrated-transformer/ 
* http://www.peterbloem.nl/blog/transformers

Naive Self-Attention
------------

Exploration notebook that anlyse the basic idea behind self attention. Notebook located in:

```
./notebooks/naive_self_attention.ipynb
```

Transformer module
------------

Prototyping notebook that describes the base transformer module. It is located in 

```
./notebooks/transformer_module.ipynb
```
Final code is implemented as python module under **./src** folder

Clasification experiemtent
------------

The experiment consist of basic sentiment anlyisis of IMDB movie comments. The exploration code is implemented only in cpu, but it is easy 
to extend it to GPU. One batch overfit is done in orfder to show the correctness of the method. Future work will include traing python script in order to train the model with GPU.

```
./notebooks/experiment_clssification.ipynb
```

Character generation experiemtent
------------
A simple autoregresive module was done with transformers in order to learn its text generation capabilities. Same future work as the previous experiment is required.

```
./notebooks/experiment_generation.ipynb
```

