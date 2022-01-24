# Deep Equilibrium Models
Repository implementing [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377).

If you use this repository, be sure to cite the authors of the original paper.
This implementation is mine, but the ideas are theirs.

Also check out [their implementation](https://github.com/locuslab/deq).

## What's a Deep Equilibrium Model?

DEQs are a form of implicit deep learning; they don't simply move data through
layers and end up at a result. At a very high level, we use a neural network to
define a vector-valued function, then we find the root of that function. The
root is our result, and we use some calculus to propagate the loss back to the
parameters of our neural network.

### Pros
- Small memory footprint

### Cons
- Slow in inference, slow in training
- Can be unstable
- Variable inference time
- Complicated

### More info:
For a very in-depth explanation, read the original paper. Also, the same authors provide a more concise and general
formulation of DEQs in their follow-up work, on [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656).

## Project Structure:

```
DeepEquilibrium
├── cifar_example.py
├── data
├── DEQ.py
├── README.md
├── rootfind.py
├── train.py
├── unit.py
└── wandb
```

## TODO List:
- [x] Trivial example working, in the forward pass (size 10)
- [x] Trivial example working in the backwards pass (learn identity or constant)
- [x] Get batching working (how does original repo do it?)
- [x] CIFAR-10 (working?)
- [ ] Use the approximate Jacobian Inverse and not the full $n \times n$ matrix
- [x] More complicated 2D and 3D inputs and outputs

## Contributing:
Anyone is welcome to open an issue or PR. I think my implementation is correct,
but if you see something let me know! I found the original paper really interesting,
but the official repo very confusing.
