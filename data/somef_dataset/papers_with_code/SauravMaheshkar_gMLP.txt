# Flax implementation of gMLP from "Pay Attention to MLPs"

![](https://github.com/SauravMaheshkar/gMLP/blob/main/assets/gMLP%20Banner.png?raw=true)

It's no news that transformers have dominated the field of deep learning ever since 2017. But, Hanxiao Liu, Zihang Dai, David R. So and Quoc V. Le in their recent work titled ["Pay Attention to MLPs"](https://arxiv.org/abs/2105.08050) propose a new architecture **gMLP** (essentially MLPs with gating) that performs as well as Transformers in key language and vision applications. Based on the comparisons shown in the paper the authors show that self-attention is **not** critical for Vision Transformers !!, as gMLP can achieve the same accuracy, thus bringing into question the validity of Attention.

This repository includes an implementation of gMLP written in [Flax](https://github.com/google/flax). Most of the codebase is inspired from [Phil Wang](https://github.com/lucidrains)'s implementations in [Pytorch](https://github.com/lucidrains/g-mlp-pytorch) and [Haiku](https://github.com/lucidrains/mlp-gpt-jax).

**NOTE: Causal Nature of Spatial Gating Unit hasn't been implemented yet**

## Installation

You can install this package from PyPI:

```sh
pip install gmlp-flax
```

Or directly from GitHub:

```sh
pip install --upgrade git+https://github.com/SauravMaheshkar/gMLP.git
```

## Usage

```python
import jax
from gmlp_flax import gMLP

random_key = jax.random.PRNGKey(0)

x = jax.random.randint(key=random_key, minval=0, maxval=20000, shape=(1, 1000))

init_rngs = {"params": random_key}

gMLP(num_tokens=20000, dim=512, depth=4).init(init_rngs,x)
```

## Development

### 1. Conda Approach

```sh
conda env create --name <env-name> sauravmaheshkar/gmlp
conda activate <env-name>
```

### 2. Docker Approach

```sh
docker pull ghcr.io/sauravmaheshkar/gmlp-dev:latest
docker run -it -d --name <container_name> ghcr.io/sauravmaheshkar/gmlp-dev
```
Use the [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension in VSCode and [attach to the running container](https://code.visualstudio.com/docs/remote/attach-container). The code resides in the `code/` dir.

Alternatively you can also download the image from [Docker Hub](https://hub.docker.com/r/sauravmaheshkar/gmlp-dev).

```sh
docker pull sauravmaheshkar/gmlp-dev
```

## Citations

```bibtex
@misc{liu2021pay,
    title   = {Pay Attention to MLPs},
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
