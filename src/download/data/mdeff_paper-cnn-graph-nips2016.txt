# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

[Michaël Defferrard](https://deff.ch),
[Xavier Bresson](https://www.ntu.edu.sg/home/xbresson),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst), \
Conference on Neural Information Processing Systems (NIPS), 2016.

> In this work, we are interested in generalizing convolutional neural networks (CNNs) from low-dimensional regular grids, where image, video and speech are represented, to high-dimensional irregular domains, such as social networks, brain connectomes or words' embedding, represented by graphs.
> We present a formulation of CNNs in the context of spectral graph theory, which provides the necessary mathematical background and efficient numerical schemes to design fast localized convolutional filters on graphs.
> Importantly, the proposed technique offers the same linear computational complexity and constant learning complexity as classical CNNs, while being universal to any graph structure.
> Experiments on MNIST and 20NEWS demonstrate the ability of this novel deep learning system to learn local, stationary, and compositional features on graphs.

```
@inproceedings{cnn_graph,
  title = {Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering},
  author = {Defferrard, Micha\"el and Bresson, Xavier and Vandergheynst, Pierre},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year = {2016},
  archiveprefix = {arXiv},
  eprint = {1606.09375},
  url = {https://arxiv.org/abs/1606.09375},
}
```

## Resources

PDF available at [arXiv], [NIPS], [EPFL].

Related: [poster], [slides], [video], [code].

[arXiv]: https://arxiv.org/abs/1606.09375
[NIPS]: https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering
[EPFL]: https://infoscience.epfl.ch/record/218985
[poster]: https://doi.org/10.5281/zenodo.1318419
[slides]: https://doi.org/10.5281/zenodo.1318411
[video]: https://youtu.be/cIA_m7vwOVQ
[code]: https://github.com/mdeff/cnn_graph

## Compilation

Compile the latex source into a PDF with `make`.
Run `make clean` to remove temporary files and `make arxiv.zip` to prepare an archive to be uploaded on arXiv.

## Figures

All the figures are in the [`figures`](figures/) folder.
PDFs can be generated with `make figures`.

## Peer-review

The paper got a [metareview](review/meta_review.htm) based on [six reviews](review/reviews.htm), on which our [rebuttal](review/rebuttal.txt) is based.
The reviews are also at [NIPS].
