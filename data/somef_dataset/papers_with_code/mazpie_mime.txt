# MIME: Maximising Information for Meta Exploration

This is the code for the MSc Project "Learning to Explore Via Meta Reinforcement Learning", submitted by the student Pietro Mazzaglia for the degree of Master of Science in ACS: Artificial Intelligence to the University of Manchester.

The repo mainly contains the implementation of MIME. a gradient-based meta-RL model augmented with strong exploration capabilities.

#### MAML (left/top) vs MIME (right/bottom) trajectories on the Dog-Feeding Robot Navigation task
<img src="images/maml_dogs.png" width="40%"> 
<img src="images/mime_dogs.png" width="40%">

## Getting started

#### Requirements
 - Python 3.6 or above (e.g. 3.6.10)
 - PyTorch 1.3.1
 - Gym 0.17.2


To avoid any conflict with your existing Python setup, I suggest to use [pyenv]('https://github.com/pyenv/pyenv') with [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/). 

Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt), with the command:

```
pip install -r requirements.txt
```


## Usage

#### Available Environments

The environment used in the MSc Project Results chapter are available through the configs file under the folder `configs/msc_project`, and they are:

**2D Navigation**

- Wide-Ring Goals
- Hard-Exploration Goals
- Close-Ring Goals
- Dog-Feeding Robot

**Swing-Up Pendulum**

- Dense
- Sparse

**Meta-World** 

- ML10


#### Training
You can use the [`train.py`](train.py) script in order to train the model:
```
python train.py  --use-vime --adapt-eta --config configs/msc_project/dog-feeding-robot-navigation.yaml --output-folder ../mime-experiments/metaworld_ml10/mime/0 --seed 0 --num-workers 6
```

The relevant flags to switch models are:

`--add-noise`: add ![noise](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D%280%2C1%29) to the rewards

`--use-vime`: activates the exploration module

`--adapt-eta`: activates the meta-learning of the ![eta](https://latex.codecogs.com/gif.latex?%5Ceta) parameter

`--e-maml`: trains using the E-MAML meta-objective reformulation

To clarify:
- **MAML** ([paper](https://arxiv.org/abs/1703.03400)): does not use any of the above flag
- **MAML+noise**: needs `--add-noise`
- **MAML+expl**: needs `--use-vime`
- **MIME**: needs both `--use-vime` and `--adapt-eta`
- **E-MAML** ([paper](https://arxiv.org/abs/1803.01118)): needs `--e-maml`


Other useful flags are:

`--seed`: to set the training seed

`--output-folder`: where to save the files

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --output-folder ../mime-experiments/metaworld_ml10/mime/4 
```

#### Visualization

Both the training and testing results are saved in the output folder indicated, providing:

- _config.json_ : containing the model training's parameters
- _policy.th_ : the saved policy model
- _dynamics.th_ : the saved dynamics model (only if `--use-vime` was used)
- _train_log.txt_ : command-line logs for training
- _train_result.csv_ : training data (returns, dynamics loss, eta, ...)
- _test_log.txt_ : command-line logs for testing
- _test_result.csv_ : testing data (returns, dynamics loss, eta, ...)
- **test**: folder containing the testing Tensorboard files
- **train_trajectories**: folder containing the Tensorboard files to visualize the trajectories (2D Navigation environments only)

To visualize Tensorboard results you can use:

```
tensorboard --logdir ../mime-experiments/ --samples_per_plugin images=100
```

#### PEARL

The environments and the config files for PEARL are available under the `pearl` folder.

The original PEARL implementation can be found [here](https://github.com/katerakelly/oyster).

## Acknowledgments

The initial MAML implementation that has been here reworked and expandend in many ways was originally developed by Tristan Deleu as a PyTorch re-implementation of MAML.

I would like to sincerely thank him for its clean and well-organised code. Its work is available at [this repository](https://github.com/tristandeleu/pytorch-maml-rl).

I also would like to thank the researchers that created the Meta-World ([paper](https://arxiv.org/abs/1910.10897)) benchmark 