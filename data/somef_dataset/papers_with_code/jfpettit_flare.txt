# flare

![**PPO agent on LunarLanderContinuous-v2**](/src/lunarlandercontinuous.gif)

*PPO agent trained to play [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/). Reward per episode at this point was ~230.*


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Details](#details)
- [Contributing](./CONTRIBUTING.md)
- [References](#references)
- [More to come](#more-to-come)

```flare``` is a small reinforcement learning library. Currently, the use case for this library is small-scale RL experimentation/research. Much of the code is refactored from and built off of [SpinningUp](https://spinningup.openai.com/en/latest/), so massive thanks to them for writing quality, understandable, and performant code.

(old) Blog post about this repository [here](https://jfpettit.svbtle.com/rlpack).

## Installation

**MPI parallelization will soon be removed. Work is being done to rebase the code using [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) which uses PyTorch's multiprocessing under the hood.**

**Flare supports parallelization via MPI!** So, you'll need to install [OpenMPI](https://www.open-mpi.org/) to run this code. [SpinningUp](https://spinningup.openai.com/en/latest/user/installation.html#installing-openmpi) provides the following installation instructions:

### On Ubuntu:

```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```

### On Mac OS X 

```
brew install openmpi
```

If using homebrew doesn't work for you, consider these [instructions](http://www.science.smith.edu/dftwiki/index.php/Install_MPI_on_a_MacBook).  

### On Windows

If you're on Windows, [here is a link to some instructions](https://nyu-cds.github.io/python-mpi/setup/).

### Installing flare

It is recommended to use a virtual env before installing this, to avoid conflicting with other installed packages. Anaconda and Python offer virtual environment systems.

Clone the repository and cd into it: 

```
git clone https://github.com/jfpettit/flare.git
cd flare
```

**The next step depends on your package manager.**

If you are using pip, pip install the ```requirements``` file:

```
pip install -r requirements.txt
```

Alternatively, if you're using Anaconda, create a new Anaconda env from the ```environments.yml``` file, and activate your new conda environment:

```
conda env create -f environment.yml
conda activate flare
```

A third option, if you don't want to clone a custom environment or run through the ```requirements.txt``` file, is to simply pip install the repository via:

```
pip install -e git+https://github.com/jfpettit/flare.git@98d6d3e74dfadc458b1197d995f6d60ef516f1ee#egg=flare
```

## Usage

### Running from command line

Each algorithm implemented can be run from the command line. A good way to test your installation is to do the following:

```
python -m flare.run
```

This will run [PPO](https://arxiv.org/abs/1707.06347) on [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) with default arguments. If you want to change the algorithm to A2C, run on a different env, or otherwise change some defaults with this command line interface, then do ```python -m flare.run -h``` to see the available optional arguments.

### Running in a Python file

Import required packages:

```python
import gym
from flare.polgrad import a2c 

env = gym.make('CartPole-v0') # or other gym env
epochs = 100
a2c.learn(env, epochs)
```

The above snippet will train an agent on the [CartPole environment](http://gym.openai.com/envs/CartPole-v1/) for 100 epochs. 

You may alter the architecture of your actor-critic network by passing in a tuple of hidden layer sizes to your agent initialization. i.e.:

```python
from flare.polgrad import ppo 
hidden_sizes = (64, 32)
ppo.learn(env, epochs=100, hidden_sizes=hidden_sizes)
```

## Details

This repository is intended to be a lightweight and simple to use RL framework, while still getting good performance.

Algorithms will be listed here as they are implemented: 

- [REINFORCE Policy Gradient (this link is to a PDF)](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [Advantage Actor Critic (A2C)](https://arxiv.org/abs/1602.01783)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477)
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290)

The policy gradient algorithms (REINFORCE, A2C, PPO), support running on multiple CPUs/GPUs via PyTorch Lightning. The Q Policy Gradient algorithms (SAC, DDPG, TD3) do not yet use Lightning, they will soon be brought up to parity with the policy gradient algorithms.

If you wish to build your own actor-critic from scratch, then it is recommended to use the [FireActorCritic](https://github.com/jfpettit/flare/blob/master/flare/kindling/neuralnets.py#L143) as a template.


## Contributing

We'd love for you to contribute! Any help is welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for contributor guidelines and info.

## References
- [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/)
- [FiredUp](https://github.com/kashif/firedup)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [A3C paper](https://arxiv.org/abs/1602.01783)
- [Pytorch RL examples](https://github.com/pytorch/examples/tree/master/reinforcement_learning)
- [PyTorch Lightning RL example](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/reinforce_learn_Qnet.py)

## More to come!
- Update Q-policy gradient algorithms to use Pytorch Lightning
- Comment code to make it clearer
- Test algorithm performance
