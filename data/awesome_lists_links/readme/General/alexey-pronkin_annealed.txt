# annealed

Implementation and reproducing paper "Improving Explorability in Variational Inference with Annealed Variational Objectives Bayesian methods" https://arxiv.org/abs/1809.01818

## Team

- Aleksei Pronkin
- Mikhail Kurenkov
- Timur Chikichev
## Proposal

see: [Proposal](Baesian_methods_project_proposal_HVI_AVO.pdf) in this repo.

## Reproducing

### Training VAE
For reproducing experiments with VAE use following commands

`
python3 run_vae.py -epoch 100 train vae_hvi
`

Epoch is a number of epoch for training. vae_hvi is a possible model. Other possible models are vae, vae_hvi, 
vae_hvi_avo. Logs is saved to lightning_logs. Checkpoint is saved to lightning_logs/versrion_{version_number}/checkpoints.

### Evaluation VAE
For testing VAE use following command

`
python run_vae.py -model-checkpoint pretrained_models/vae_hvi_avo.ckpt test vae_hvi_avo
`

This command also generate generation.png and reconstruction.png
## Goal

During this project we are going to implement methods and repeat experiments of this paper. https://arxiv.org/abs/1809.01818

## Experiments

- Biased noise model.
- Toy energy fitting.
- Quantitative analysis on robustness to beta annealing.
- Amortized inference on MNIST dataset.

## Results

Annealed toy distributions:

![alt text](report/images/transitions_targets_abc.png "Annealed target distributions")

Toy distributions experiments:

![alt text](toy_figs.png "Toy experiments")

Reconstruction of VAE. Left is ground truth, right is reconstructed.

![alt text](report/images/reconstruction.png "VAE reconstructions")

## Models

- Variational Auto Encoder (VAE) https://arxiv.org/abs/1312.6114 with parameters from https://github.com/jmtomczak/vae_householder_flow
- Hierarchical Variational Model (HVM) based on https://arxiv.org/abs/1511.02386.
- Importance Weighted Auto-Encoder (IWAE) based on https://arxiv.org/abs/1509.00519.

## Pipeline

- Implement models.
- Implement AVO method https://arxiv.org/abs/1809.01818.
- Implement Hierarchical Variational Inference (HVI) method https://arxiv.org/abs/1511.02386.
- Analyze a behavior of AVO objective on toy examples.
- Conduct experiments with VAE and AVO on the MNIST and CelebA datasets.

We planned to use and rewrite some code from https://github.com/joelouismarino/iterative_inference/, https://github.com/jmtomczak/vae_householder_flow, https://github.com/AntixK/PyTorch-VAE, https://github.com/haofuml/cyclical_annealing and https://github.com/ajayjain/lmconv. (We assume, that first two repositories were used in the original paper closed source code)

