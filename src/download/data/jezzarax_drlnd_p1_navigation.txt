Solution to the first project of the deep reinforcement learning nanodegree at Udacity.

## Problem definition

The reinforcement learning agent is travelling through a 2d space filled with blue and yellow bananas. The agent is expected to gather the banana if it is yellow or avoid the blue ones. The agent receives a positive reward for every yellow banana it gathers and a negative reward for every blue banana. The state the agent receives comprises of its speed as well as the raytraced positions of the nearest bananas in the field of view, the size of the state space is 37. The agent is able to move forwards and backwards as well as turn left and right, thus the size of the action space is 4. The minimal expected performance of the agent after training is a score of +13 over 100 consecutive episodes.

## Usage

### Preparation

Please ensure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed. Clone the repository and use `pipenv --three install` to create yourself an environment to run the code in. Otherwise just install the packages mentioned in Pipfile.

Due to the transitive dependency to tensorflow that comes from unity ml-agents and the [bug](https://github.com/pypa/pipenv/issues/1716) causing incompatibility to jupyter you might want to either drop the jupyter from the list of dependencies or run `pipenv --three install --skip-lock` to overcome it.

To activate a virtual environment with pipenv issue `pipenv shell` while in the root directory of the repository.

After creating and entering the virtual environment you need to set a `DRLUD_P1_ENV` shell environment which must point to the binaries of the Unity environment. Example of for Mac OS version of binaries it might be 
```
DRLUD_P1_ENV=../deep-reinforcement-learning/p1_navigation/Banana.app; export DRLUD_P1_ENV
```

Details of downloading and setting of the environment are described in Udacity nanodegree materials.

### Training (easy way)

Just follow the [training notebook](Training.ipynb).

### Training (proper way)

The executable part of code is built as a three-stage pipeline comprising of
* training pipeline
* analysis notebook
* demo notebook

The training pipeline was created with the idea of helping the researcher to keep track of his experimentation process as well as keeping the running results. The training process is spawned by executing the `trainer.py` script and is expected to be idempotent to the training results, i.e. if the result for a specific set of hyperparameters already exists and persisted, the trainer will skip launching a training session for this set of hyperparameters.

The sets of hyperparameters for training are defined inside of `trainer.py` in the `simulation_hyperparameter_reference` dictionary which is supposed to be append-only in order to keep consistency of the training result data. Each of the hyperparameters sets will produce a file with a scores of number of runs of an agent which will be stored inside of `./hp_search_results` directory with an id referring to the key from the `simulation_hyperparameter_reference` dictionary. The neural networks weights for every agent training run will be stored in the same directory with the relevant hyperparameters key as well as random seed used.

To train an agent with a new set of hyperparameters just add an item into `simulation_hyperparameter_reference` object. Here's an example of adding an item with id `25` after existing hyperparameter set with id `24`:

```
simulation_hyperparameter_reference = {
###
###  Skipped some items here
###

    24: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dueling"   ),
    25: hparm(5e-4, 4,  64,  int(1e5), 0.99, 1e-3, 10,  36, "dueling"   )
}
```

The set of hyperparameters is represented as an instance of a namedtuple `hparm` which has the following set of fields:

* lr
* update_rate
* batch_size 
* memory_size
* gamma 
* tau 
* times
* hidden_layer_size 
* algorithm

The `algorithm` field defines an implementation of an agent, currently only the following values are supported:

* `dqn` which implements a basic deep Q-learning from this [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).
* `ddqn` for double deep Q-learning from [arXiv:1509.06461](https://arxiv.org/abs/1509.06461).
* `dueling` for dueling Q-Network from [arXiv:1511.06581](https://arxiv.org/abs/1511.06581) paper.

The agents rely on a network with a single hidden layer the number of neurons for which is defined by the `hidden_layer_size` parameter.

The meaning and effects of other values for these field are discussed in the [hyperparameter search notebook](Training_hyperparameter_search_analysis.ipynb). 

## Implementation details

Two neural network architectures are defined in the `qnetwork.py` file. 
* QNetwork class implement a three-layer neural network with a parameterized hidden layer size.
* DuelQNetwork class implements a dueling q-network as described in the "Dueling Network Architectures for Deep Reinforcement Learning" paper ([arXiv:1511.06581](https://arxiv.org/abs/1511.06581))

Implementations of DQN and DDQN agents are located inside of `agents.py`. Both of them rely on the same neural network architecture as well as the replay buffer which is in `replay_buffer.py`.

To see the performance of agents using DQN and DDQN with different sets of hyperparameters (lr, batch_size, etc) as well training code example please check the [hyperparameter search notebook](Training_hyperparameter_search_analysis.ipynb).

## Results 

Please check the [following notebook](Report.ipynb) for the best set of hyperparameters I managed to identify.

