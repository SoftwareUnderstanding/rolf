# DISCERN: Diversity-based Selection of Centroids for k-Estimation and Rapid Non-stochastic Clustering
This repository contains the implementation of [DISCERN](https://link.springer.com/article/10.1007/s13042-020-01193-5) in Python. You can download the manuscript from [my website](https://alihassanijr.com/files/DISCERN.pdf) or [arXiv](https://arxiv.org/abs/1910.05933).

## :red_circle: GPU-based DISCERN
:red_circle: New :red_circle:

This is just another implementation of DISCERN, except it is implemented in PyTorch instead of your average numpy. As a result, we can use all of the CUDA greatness to boost DISCERN beyond imagination.

```python3
from DISCERN import TorchDISCERN

di = DISCERN()
di.fit(X)

clustering_labels = di.labels_
cluster_centers = di.cluster_centers_
sse_loss = di.inertia_
```
For now, it automatically sets the variables to CUDA if it is available.

For those who've read the paper: the similarity precomputation and diversity-based selection are now a lot faster than on an average CPU. I'm also working on a torch-based K-Means to really push this over the edge.

Here's the progress so far:

:white_check_mark: Cosine similarity matrix computation

:white_check_mark: Diversity-based selection

:black_square_button: Finite differences

:black_square_button: K-Estimation

:black_square_button: K-Means

## Examples
```python3
X = load_data() # This is assumed to be a 2-dimensional numpy array, where rows represent data samples.
```
Basic DISCERN instance (numpy-based):
```python3
from DISCERN import DISCERN

di = DISCERN()
di.fit(X)

clustering_labels = di.labels_
cluster_centers = di.cluster_centers_
sse_loss = di.inertia_
```
Fix the number of clusters to a specific number (only use DISCERN to initialize K-Means)
```python3
di = DISCERN(n_clusters=K)
```
Use Spherical K-Means
```python3
di = DISCERN(metric='cosine')
```
Specify an upper bound for the number of clusters
```python3
di = DISCERN(max_n_clusters=1000)
```


## Notebooks

Two Jupyter notebooks are also provided in this repository (see `examples/`). Multivariate applies DISCERN to two of the multivariate datasets in the paper.
The other (ImageNette) applies it to <a href="https://github.com/fastai/imagenette">one of the image datasets in the paper, ImageNette</a>. However, unlike the paper, the notebook uses MoCo <a href="#moco">[1]</a><a href="#mocov2">[2]</a> instead of a labeled-imagenet pretrained ResNet.

Stay tuned for more notebooks.

## Citation
```
@article{hassani2020discern,
	title        = {DISCERN: diversity-based selection of centroids for k-estimation and rapid non-stochastic clustering},
	author       = {Hassani, Ali and Iranmanesh, Amir and Eftekhari, Mahdi and Salemi, Abbas},
	year         = 2020,
	journal      = {International Journal of Machine Learning and Cybernetics},
	doi          = {10.1007/s13042-020-01193-5}
}
```

## References

<div id="moco">
[1] He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. "Momentum contrast for unsupervised visual representation learning." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729-9738. 2020. (<a href="https://arxiv.org/abs/1911.05722">arXiv</a> | <a href="https://github.com/facebookresearch/moco/">GitHub</a>) 
</div>
<div id="mocov2">
[2] Chen, Xinlei, Haoqi Fan, Ross Girshick, and Kaiming He. "Improved baselines with momentum contrastive learning." <a href="https://arxiv.org/abs/1911.05722">arXiv preprint arXiv:2003.04297</a> (2020).
</div>
