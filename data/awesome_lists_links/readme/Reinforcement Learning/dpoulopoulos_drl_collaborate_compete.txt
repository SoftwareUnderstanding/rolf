# Unity-Technologies Collaboration-Competition Project

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behaviour (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

In this project, we develop two reinforcement learning agents to collaborate in a table tennis game, so as to keep the ball in the game as long as possible. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. To solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drl_cc python=3.6
	source activate drl_cc
	```
	- __Windows__:
	```bash
	conda create --name drl_cc python=3.6
	activate drl_cc
	```

2. Install Dependencies

    - Install Pytoch by following the instructions for your system [here](https://pytorch.org/)

    - To install the necessary dependencies run `pip install ./python`

3. Download the Unity Environment

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl_cc` environment.  
```bash
python -m ipykernel install --user --name drl_cc--display-name "drl_cc"
```

5. Before running code in a notebook, change the kernel to match the `drl_cc` environment by using the drop-down `Kernel` menu.

## Usage

Open the `tennis.ipynb` on a notebook and run the cells. In any case, the weights of a pretrained network are saved in `actor_solution.pth` for the Actor network and `critic_solution_0.pth`, `critic_solution_1.pth` for the Critic network, so you can witness how a trained agent behaves. In reality, you only need the actor network weights.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## References

1. [Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).](https://arxiv.org/abs/1509.02971)

## License
[MIT](https://choosealicense.com/licenses/mit/)
