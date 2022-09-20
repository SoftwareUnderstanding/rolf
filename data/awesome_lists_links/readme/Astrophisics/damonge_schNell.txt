# schNell

schNell is a very lightweight python module that can be used to compute basic map-level noise properties for generic networks of gravitational wave interferometers. This includes primarily the noise power spectrum  "N_ell", but also other things, such as antenna patterns, overlap functions, inverse variance maps etc.

## Installation
You can install schnell simply by typing
```
pip install schnell [--user]
```
(use `--user` if you don't have admin privileges on your machine).
Or for development versions you can download the repository with git and install from there using `python setup.py install [--user]`.

## Documentation
Documentation can be found on [readthedocs](https://schnell.readthedocs.io/en/latest/).

This example [notebook](https://github.com/damonge/schNell/blob/master/examples/Nell_example.ipynb) on github also showcases the main functionalities of the module.

Check out the following videos showing the scanning patterns of different GW networks.
- [LIGO instantaneous sensitivity](https://youtu.be/54WBdWgBO8k)
- [LIGO cumulative sensitivity](https://youtu.be/ByrEqpIrQzY)
- [LISA instantaneous sensitivity](https://youtu.be/8d6gEGlboz8)

## License and credits
If you use schNell, we kindly ask you to cite its [companion paper](https://arxiv.org/abs/2005.03001).

The code is available under the [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.

If you have a problem you've not been able to debug, or a feature request/suggestion, please open an issue on github to discuss it.
