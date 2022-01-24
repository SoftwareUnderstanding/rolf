# The Limitations of Large Width in Neural Networks: A Deep Gaussian Process Perspective

This repo contains the code to reproduce the experiments in Section 6 of "The Limitations of Large Width in Neural Networks: A Deep Gaussian Process Perspective" (NeurIPS, 2021). To cite:

```
@inproceedings{pleiss2021limitations,
  title={The Limitations of Large Width in Neural Networks: A Deep {G}aussian Process Perspective},
  author={Pleiss, Geoff and Cunningham, John P},
  booktitle={Neural Information Processing Systems},
  year={2021}
}
```

![3 layer Deep GP/Bayesian NN performance as a function of width](https://github.com/gpleiss/limits_of_large_width/files/7500639/dgp3l_width.pdf)

The purpose of these experiments is 

## Requirements
- pytorch (>= 1.8.1)
- gpytorch (>= 1.5)
- tqdm
- pandas
- scipy
- fire

## Training Bayesian models (Section 6.1)
For all the experiments in Section 6.1, we perform a 2 (or 3) step pipeline.
We first compute the hyperparameters of the limiting GP,
and then we perform NUTS sampling of the Deep GP and NN posteriors at various widths.

Steps 1.5 and Step 2 should be performed for various widths and various depths to reproduce the plots in Section 6.1.

### Step 1: Train the limiting GP models
We will use these models to get the hyperparameters for the deeper models.

```sh
python runner.py new --save <LIM_GP_SAVE_NAME> --data <PATH_TO_DATA_DIR> --dataset <DATASET> --model <model_name> --n 1000 \
- train - test - kernel_fit - done
```

Options for `<model_name>` (different GP covariance types):
- `gp` (GP that moment matches a 2 layer RBF + RBF Deep GP)
- `gp_3l` (GP that moment matches a 3 layer RBF + RBF + RBF Deep GP)
- `gp_acos` (GP with an arccos kernel - moment matches a 2 layer NN)
- `gp_acos3` (GP that moment matches a 3 layer NN)

### Step 1.5 (Deep GP only!): Run SVI to initialize the Deep GP NUTS sampler
In this intermediate step, we fit a Deep GP to the data using SVI.
We will use this approximate posterior as initialization for the NUTS sampler.

```sh
python runner.py new --save <VI_SAVE_NAME> --data <PATH_TO_DATA_DIR> --dataset <DATASET> --model deep_gp --n 1000 \
- --width <WIDTH> \
- initialize <LIM_GP_SAVE_NAME> \
- train - test - kernel_fit - done
```

### Step 2: Perform NUTS sampling from the Deep GP/Bayesian NN posterior
This computes HMC samples from the true Deep GP/Bayesian NN posterior (without any approximations).

```sh
python runner.py new --save <SAVE_NAME> --data <PATH_TO_DATA_DIR> --dataset <DATASET> --model <model_name> --n 1000 \
- --width <WIDTH> \
- initialize <LIM_GP_SAVE_NAME or VI_SAVE_NAME> \
- train - test - kernel_fit - done
```

(The argument to `initialize` should be `VI_SAVE_NAME` for Deep GP models, and `LIM_GP_SAVE_NAME` for NN.)

Options for `<model_name>`:
- `deep_gp_hmc` (2 layer RBF + RBF Deep GP)
- `deep_gp_3l3l_hmc` (3 layer RBF + RBF + RBF Deep GP)
- `nn_hmc` (2 layer NN)
- `nn_3l_hmc` (3 layer NN)




## Training non-Bayesian NN (Section 6.2)
These experiments require only a single step.
Each experiment should be repeated for multiple widths to reproduce the plots in Section 6.2.

### Training MLP on MNIST
These experiments use a random hyperparameter search.
Each model from the random hyperparameter search is stored in a separate folder.
It is up to the user to find the model that produces the best hyperparameters.

```sh
python multi_runner_mnist.py --n 50000 --width ${WIDTH} --std ${STD} --seed ${SEED}
```

- std corresponds to the standard deviation of the Gaussian prior on the parameters. It is inversely proportional to the L2 regularization coefficient.

### Training ResNet models on CIFAR10
For ResNet models, we use the standard hyperparameters defined in the [original paper](https://arxiv.org/abs/1512.03385).
These hyperparameters [have been shown to work well on both narrow and wide models](https://arxiv.org/abs/1605.07146).

```sh
python runner.py new --save <SAVE_NAME> --data <PATH_TO_DATA_DIR> --dataset cifar10 --model resnet \
- --depth <DEPTH> --width <WIDTH> \
- train - test - done
```

- Options for `depth`: 8, 14
