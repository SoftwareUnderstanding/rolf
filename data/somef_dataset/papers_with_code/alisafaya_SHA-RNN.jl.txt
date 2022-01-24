# SHA-RNN

Implementation of Single Headed Attention - Recurrent Neural Networks in [Julia](https://julialang.org/) and [Knet](https://github.com/denizyuret/Knet.jl).

Stephan Merity. **Single Headed Attention RNN: Stop Thinking With Your Head**. _arXiv preprint arXiv:1911.11423_, 2019.

https://arxiv.org/abs/1911.11423v2


![SHA-RNN Model](https://raw.githubusercontent.com/alisafaya/SHA-RNN.jl/master/SHA-RNN(2).png)


After downloading the data and preprocessing it using

```bash
sh getdata.sh
```

You can train the main model of SHA-RNN paper by either:

_running [sharnn-main.jl](examples/sharnn-main.jl) in shell_

```bash
cd examples
julia sharnn-main.jl
```

_or using [SHA-RNN](notebooks/SHA-RNN.ipynb) notebook_.

This implementation is identical to the one of Smerity's original implementation [sha-rnn](https://github.com/Smerity/sha-rnn). 

But it is slower, since it does not use the same performance tricks that the version of SHA-RNN that was implemented using pytorch uses.


### Features to be added to get faster training :

- Fused layer normalization (check if [Apex](https://github.com/NVIDIA/apex/) CUDA code can be used with Knet)
- Using half precision floating point (Float16) for memory efficiency
- Checkpoint feature similar to pytorch's [checkpoint](https://pytorch.org/docs/stable/checkpoint.html).
