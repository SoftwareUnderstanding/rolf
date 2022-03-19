# MLP Mixer Pytorch

Pytorch implementation of [MLP-Mixer](https://arxiv.org/abs/2105.01601).

<img src="https://miro.medium.com/max/2400/1*DqrznEKzR_xB-CEhpOav3A.png" />


## Sample usage

```console
foo@bar:❯ pip install mlp_mixer
```

```Python
from mlp_mixer import MLPMixer

model = MLPMixer(
        img_size=IMG_SZ,
        img_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        mixer_depth=DEPTH,
        num_patches=NUM_PATCHES,
        num_channels=NUM_CHANNELS,
        expansion=EXPANSION,
        dropout=DROPOUT,
    )
```


## Citations

```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```