# PPO
An Implementation of Proximal Policy Optimization. Based off of Schulman et. al's Paper (https://arxiv.org/pdf/1707.06347.pdf) and OpenAI Baselines (https://github.com/openai/baselines)

Three useful aspects of this project are:

1. Simplicity. Almost of the code in this repository is original. The only code taken directly from OpenAI Baselines is the atari_wrappers.py file and few weight initialization methods in the networks.py file. Since the sole purpose of this repository is to be PPO algorithm, it will hopefully be more easy to use than the Baselines implementation, which is more broad in scope. Moreover, early tests have demonstrated near-competitive performance with the Baselines implementation.

2. Out of the box Tensorboard Integration. This repository has Tensorboard Integration for loss values, rewards, and more. Simply add a command line flag to indicate you would like the results to be logged to Tensorboard.

3. Optimizers file. The optimizers.py file has support to easily add additional optimizer methods. Adam is used by default, but, if one wanted to create a new optimizer using a different algorithm this can be easily done.

# Limitations
This repository does not support parellizing actors, which can make it slower than Baselines. It also does not fully integrate the OpenAI wrappers for Mujocu and other non-Atari environments.

# Getting Started
Python 3.5+ is recommended. Install the dependencies with the following command:
```
pip install -r requirements.txt
```

Then, I recommend training a model in a simple environment, like Cartpole. Use this command
to do that. This will also log the results to tensorboard and save the model (Training takes about 5 min. on my 2018 Macbook Pro):
```
py runner.py --env-id='CartPole-v1' --learning-rate='lambda x: x * 1e-4' --shared-network='fc3'  --num-batches=500 --tb-path='./tbs/CartPole-v1/' --log-every=1 --save-path='./models/CartPole-v1/' --save-every=50
```

Use this command to watch your trained model play:
```
py runner.py --env-id='CartPole-v1' --mode='test' --restore-path='./models/CartPole-v1/-500' --shared-network='fc3'
```

Launch your Tensorboard:
```
tensorboard --logdir='./tbs/CartPole-v1/'
```

# Train Pong
The following command trains an agent to achieve near perfect scores on Pong. A pre-trained model and tensorboard can be found in the trained_models directory.
```
python runner.py --env-id='PongNoFrameskip-v4' --shared-network='cnn' --learning-rate="lambda x: x * 3e-4" --num-batches=10000 --env-steps=128 --num-envs=4 --eps=0.1 --tb-path="./tbs/PongNoFrameskip-v4/" --save-path="./models/PongNoFrameskip-v4/" --log-every=1 --save-every=500
```

# Train Space Invaders
The following command trains a Space Invaders agent. A pre-trained model and tensorboard can be found in the trained_models directory. While this agent will be very good at Space Invaders, training for additional time (maybe 40000 batches) will likely create an even better one.
```
python runner.py --env-id='SpaceInvadersNoFrameskip-v4' --shared-network='cnn_lstm' --learning-rate="lambda x: x * 3e-4" --num-batches=20000 --env-steps=128 --num-envs=8 --eps=0.1 --tb-path="./tbs/SpaceInvadersNoFrameskip-v4/" --save-path="./models/SpaceInvadersNoFrameskip-v4/" --log-every=1 --save-every=500
```

# Hyperparameter Tips
Annealing the learning rate is very important if you would like to converge on an optimal solution (not annealing will usually result in an unstable model even if it performs well for a short time). Playing with multiple environments is also important. 4+ is usually needed.

You may find more information on hyperparameters in the Schulman PPO Paper and Stooke & Abbeel's paper: https://arxiv.org/pdf/1803.02811.pdf

# Final Notes
I hope you find this useful or at least enjoyable. Also, I am sure there are ways this project can be improved! I am happy to take constructive feedback into account.
