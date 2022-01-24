# Predictive Collective Variable Discovery with Deep Bayesian Models

[Markus Schöberl](https://github.com/mjschoeberl), [Nicholas Zabaras](https://www.zabaras.com), [Phaedon-Stelios Koutsourelakis](http://www.contmech.mw.tum.de)

Python/PyTorch implementation of the discovery of collective variables (CV) in atomistic systems.
Extending spatio-temporal scale limitations of models for complex atomistic systems considered in biochemistry and materials science necessitates the development of enhanced sampling methods. The potential acceleration in exploring the configurational space by enhanced sampling methods depends on the choice of collective variables (CVs). This software implements the discovery of CVs as a Bayesian inference problem and considers CVs as hidden generators of the full-atomistic trajectory. The ability to generate samples of the fine-scale atomistic configurations using limited training data allows to compute estimates of observables as well as our probabilistic confidence on them. The formulation is based on emerging methodological advances in machine learning and variational inference. The discovered CVs are related to physicochemical properties which are essential for understanding mechanisms especially in unexplored complex systems. We provide a quantitative assessment of the CVs in terms of their predictive ability for alanine dipeptide (ALA-2) and ALA-15 peptide.

Results of the following publication are based on this implementation: [Predictive Collective Variable Discovery with Deep Bayesian Models](https://doi.org/10.1063/1.5058063).


## Dependencies
- Python 2.7.9
- PyTorch 0.3.0
- Scipy
- Matplotlib
- Imageio
- Future


## Installation
- Install PyTorch and other dependencies.
- Clone this repo with:
```
git clone https://github.com/cics-nd/predictive-cvs.git
cd predictive-cvs
```


## Dataset
The datasets for ALA-2 and ALA-15 are located in the subfolders `./data_peptide/ala-2/.` and `./data_peptide/ala-15/.`,
respectively. In both cases, scripts for running the original molecular dynamic (MD) simulation leading to the reference trajectory are placed in `./data_peptide/<peptide-name>/gromacs/.`. `Gromacs 5.0.4` has been used for performing the MD simulations. For further details regarding MD simulations, please see http://www.gromacs.org.

### ALA-2
For alanine dipeptide, prepared datasets with N=[50, 100, 200, 500] samples as used in the corresponding publication are provided.

### ALA-15
For alanine 15 peptide, we provide datasets with N=[300, 1500, 3000, 5000].

## Training

### MAP Estimate 

Train the model and obtain a MAP estimate of the predicted trajectory. With the `--dataset` option one is able to choose N, the amount of samples considered for training the model and the peptide (i.e. ALA-2 or ALA-15). For ALA-2, use `ma_<N>` while `<N>` is replaced by N=[50, 100, 200, 500].
For ALA-15, use `m_<N>_ala_15` and N=[300, 1500, 3000, 5000].
```
python main.py --dataset ma_200 --epoch 8000 --batch_size 64 --z_dim 2 --samples_pred 1000 --ard 1.0e-5 
```
Files written during training and the predicted trajectory are stored in `./results/<dataset>/<date_time>/.`.
The file `samples_aevb_<dataset>_z_<dim>_<batch-size>_<max-epoch>.txt` contains samples <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\sim%20p(\boldsymbol{x}|\boldsymbol{\theta}_{\text{MAP}})" border="2"/> where each column represents one sample <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}_i" border="2">.
Aforementioned command produces 1000 samples <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\sim%20p(\boldsymbol{x}|\boldsymbol{\theta}_{\text{MAP}})" border="2"/>. This is adjusted by setting `--samples_pred`.

### Uncertainty quantification utilizing approximate posterior inference

The following command trains the model and produces a MAP estimate of the predicted trajectory. Additionally the approximate posterior distribution of the decoding network parameters <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}" border="1"/> is estimated.
```
python main.py --dataset ma_200 --epoch 8000 --batch_size 64 --z_dim 2 --samples_pred 1000 --ard 1.0e-5 
--npostS 500
```
This command trains the model parametrization and produces a MAP estimate stored as `samples_aevb_<dataset>_z_<dim>_<batch-size>_<max-epoch>.txt`, using <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_{\text{MAP}}" border="1"/>. In addition, the posterior <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_i\sim\%20p(\boldsymbol{\theta}|\boldsymbol{X})" border="2"/> is estimated by employing Laplace's approximation.
The aforementioned command produces 500 posterior samples of the decoding network parameters
<img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_i\sim\%20p(\boldsymbol{\theta}|\boldsymbol{X})" border="2"/>. One is able to control the amount of produced posterior samples by changing `--npostS 500`.
For each <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_i" border="2"/>, 1000 samples (`--samples_pred 1000`) <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\sim%20p(\boldsymbol{x}|\boldsymbol{\theta}_{\text{i}})" border="2"/> are predicted.

## Citation

Please consider to cite the following work:
```latex
@article{schoeberl2019,
author = {Schöberl,Markus  and Zabaras,Nicholas  and Koutsourelakis,Phaedon-Stelios },
title = {Predictive collective variable discovery with deep Bayesian models},
journal = {The Journal of Chemical Physics},
volume = {150},
number = {2},
pages = {024109},
year = {2019},
doi = {10.1063/1.5058063},
URL = {https://doi.org/10.1063/1.5058063},
eprint = {https://doi.org/10.1063/1.5058063}
}
```

## Acknowledgments

The code is inspired by the implementation of Kingma and Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

## Questions

Please feel free to contact Markus Schöberl (mschoeberl@gmail.com) for questions and comments.
