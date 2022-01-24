
# Expectigrad: Fast Stochastic Optimization with Robust Convergence Properties
![pypi](https://img.shields.io/badge/pypi-0.0.0-blue)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
![python](https://img.shields.io/badge/python-3.5-blue)

[![pytorch](https://img.shields.io/badge/pytorch-yes-brightgreen)](#pytorch)
[![tensorflow1](https://img.shields.io/badge/tensorflow%201-yes-brightgreen)](#tensorflow-1.x)
[![tensorflow2](https://img.shields.io/badge/tensorflow%202-yes-brightgreen)](#tensorflow-2.x)

Expectigrad is a first-order stochastic optimization method that fixes the
[known divergence issue](https://arxiv.org/abs/1904.09237)
of Adam, RMSProp, and related adaptive methods while offering better performance on
well-known deep learning benchmarks.

Expectigrad introduces two innovations to adaptive gradient methods:
- **Arithmetic RMS:** Computes the true RMS instead of an exponential moving average (EMA).
This makes Expectigrad more robust to divergence and, in theory, less susceptible to
gradient noise.
- **Outer momentum:** Applies momentum _after_ adapting the step sizes, not
before.
This reduces bias in the updates by preserving the
[superposition property](https://en.wikipedia.org/wiki/Superposition_principle).

See [the paper](https://arxiv.org/abs/2010.01356) for more details.

Pytorch, TensorFlow 1.x, and TensorFlow 2.x are all supported.
See [installation](#installation) and [usage](#usage) below to get started.

### Pseudocode

> ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7BInitialize%20network%20parameters%7D%5C%20x) <br/>
> ![equation](https://latex.codecogs.com/svg.latex?s%20%5Cgets%200) <br/>
> ![equation](https://latex.codecogs.com/svg.latex?n%20%5Cgets%200) <br/>
> ![equation](https://latex.codecogs.com/svg.latex?m%20%5Cgets%200) <br/>
> ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Bfor%7D%5C%20t%3D1%2C2%2C%5Cldots%5C%20%5Ctext%7Buntil%20convergence%20do%7D) <br/>
>  ![equation](https://latex.codecogs.com/svg.latex?%5Cquad%20g%20%5Cgets%20%5Cnabla%20f%28x%29) <br/>
>  ![equation](https://latex.codecogs.com/svg.latex?s%20%5Cgets%20s%20&plus;%20g%5E2) <br/>
>  ![equation](https://latex.codecogs.com/svg.latex?n%20%5Cgets%20n%20&plus;%20%5Ctext%7Bsign%7D%28g%5E2%29) <br/>
>  ![equation](https://latex.codecogs.com/svg.latex?m%20%5Cgets%20%5Cbeta%20m%20&plus;%20%281-%5Cbeta%29%20%5Ccdot%20%5Cfrac%7Bg%7D%7B%5Cepsilon%20&plus;%20%5Csqrt%7B%5Cfrac%7Bs%7D%7Bn%7D%7D%7D) <br/>
>  ![equation](https://latex.codecogs.com/svg.latex?x%20%5Cgets%20x%20-%20%5Cfrac%7B%5Calpha%7D%7B1-%5Cbeta%5Et%7D%20%5Ccdot%20m) <br/>
> ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Bend%20for%7D) <br/>
> ![eqaution](https://latex.codecogs.com/svg.latex?%5Ctext%7Breturn%7D%5C%20x)

### Citing

If you use this code for published work, please cite [the original paper](https://arxiv.org/abs/2010.01356):

```
@article{daley2020expectigrad,
  title={Expectigrad: Fast Stochastic Optimization with Robust Convergence Properties},
  author={Daley, Brett and Amato, Christopher},
  journal={arXiv preprint arXiv:2010.01356},
  year={2020}
}
```

---

## Installation

Use pip to quickly install Expectigrad:

```
pip install expectigrad
```

Or you can clone this repository and install manually:

```
git clone https://github.com/brett-daley/expectigrad.git
cd expectigrad
python setup.py -e .
```

## Usage

Pytorch and both versions of TensorFlow are supported.
Refer to the code snippets below to instantiate the optimizer for your deep learning framework.

### Pytorch

```python
import expectigrad

expectigrad.pytorch.Expectigrad(
    params, lr=0.001, beta=0.9, eps=1e-8, sparse_counter=True
)
```

| Args | | |
| --- | :-: | --- |
| params | (`iterable`) | Iterable of parameters to optimize or dicts defining parameter groups. |
| lr | (`float`) | The learning rate, a scale factor applied to each optimizer step. Default: `0.001` |
| beta | (`float`) | The decay rate for Expectigrad's bias-corrected, "outer" momentum. Must be in the interval [0, 1). Default: `0.9` |
| eps | (`float`) | A small constant added to the denominator for numerical stability. Must be greater than 0. Default: `1e-8` |
| sparse_counter | (`bool`) | If True, Expectigrad's counter increments only where the gradient is nonzero. If False, the counter increments unconditionally. Default: `True` |

---

### Tensorflow 1.x

```python
import expectigrad

expectigrad.tensorflow1.ExpectigradOptimizer(
    learning_rate=0.001, beta=0.9, epsilon=1e-8, sparse_counter=True,
    use_locking=False, name='Expectigrad'
)
```

| Args | | |
| --- | :-: | --- |
| learning_rate | | The learning rate, a scale factor applied to each optimizer step. Can be a float, `tf.keras.optimizers.schedules.LearningRateSchedule`, `Tensor`, or callable that takes no arguments and returns the value to use. Default: `0.001` |
| beta | (`float`) | The decay rate for Expectigrad's bias-corrected, "outer" momentum. Must be in the interval [0, 1). Default: `0.9` |
| epsilon | (`float`) | A small constant added to the denominator for numerical stability. Must be greater than 0. Default: `1e-8` |
| sparse_counter | (`bool`) | If True, Expectigrad's counter increments only where the gradient is nonzero. If False, the counter increments unconditionally. Default: `True` |
| use_locking | (`bool`) | If True, apply use locks to prevent concurrent updates to variables. Default: `False` |
| name | (`str`) | Optional name for the operations created when applying gradients. Default: `'Expectigrad'` |

---

### Tensorflow 2.x

```python
import expectigrad

expectigrad.tensorflow2.Expectigrad(
    learning_rate=0.001, beta=0.9, epsilon=1e-8, name='Expectigrad', **kwargs
)
```

| Args | | |
| --- | :-: | --- |
| learning_rate | | The learning rate, a scale factor applied to each optimizer step. Can be a float, `tf.keras.optimizers.schedules.LearningRateSchedule`, `Tensor`, or callable that takes no arguments and returns the value to use. Default: `0.001` |
| beta | (`float`) | The decay rate for Expectigrad's bias-corrected, "outer" momentum. Must be in the interval [0, 1). Default: `0.9` |
| epsilon | (`float`) | A small constant added to the denominator for numerical stability. Must be greater than 0. Default: `1e-8` |
| sparse_counter | (`bool`) | If True, Expectigrad's counter increments only where the gradient is nonzero. If False, the counter increments unconditionally. Default: `True`
| name | (`str`) | Optional name for the operations created when applying gradients. Default: `'Expectigrad'` |
| **kwargs | | Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm` is gradient clipping by norm; `clipvalue` is gradient clipping by value; `decay` is included for backward compatibility to allow time inverse decay of learning rate; `lr` is included for backward compatibility, recommended to use `learning_rate` instead. |

---
