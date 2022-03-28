# Flax Implementation of ResMLP

![](https://github.com/SauravMaheshkar/ResMLP-Flax/blob/main/assets/ResMLP%20Banner.png?raw=true)

In the past few months there have been various papers proposing MLP based architectures without Attention or Convolutions In this [paper](https://arxiv.org/abs/2105.03404), titled 'ResMLP: Feedforward networks for image classification with data-efficient training', the authors introduce a new architecture for efficient image classification.

## Installation

You can install this package from PyPI:

```sh
pip install resmlp-flax
```

Or directly from GitHub:

```sh
pip install --upgrade git+https://github.com/SauravMaheshkar/ResMLP-Flax.git
```

## Usage

```python
from resmlp_flax.model import ResMLP

model = ResMLP(dim=512, depth=10, patch_size=16, num_classes=10)
```

## Development

### 1. Conda Approach

```sh
conda env create --name <env-name> sauravmaheshkar/resmlp
conda activate <env-name>
```

### 2. Docker Approach

```sh
docker pull ghcr.io/sauravmaheshkar/resmlp-dev:latest
docker run -it -d --name <container_name> ghcr.io/sauravmaheshkar/resmlp-dev
```

Use the [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension in VSCode and [attach to the running container](https://code.visualstudio.com/docs/remote/attach-container). The code resides in the `code/` dir.

Alternatively you can also download the image from [Docker Hub](https://hub.docker.com/r/sauravmaheshkar/resmlp-dev).

```sh
docker pull sauravmaheshkar/resmlp-dev
```

## Citations

```bibtex
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and Edouard Grave and Gautier Izacard and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
