# demo-A2C

A demo of the discrete action space advantage actor critic (A2C) (Mnih et al. 2016). 

The animation below shows the learned behavior on `CartPole-v0`. The goal is to keep the pole upright. For comparison, here's <a href="https://gym.openai.com/docs/#environments">a random policy</a>. 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-CartPole-v0.gif" width=500>

Here's the learning curve: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/lc-CartPole-v0.png" width=450>


### How to use: 

The dependencies are: `pytorch`, `gym`, `numpy`, `matplotlib`, `seaborn`. The lastest version should work. 

For training (the default environment is `CartPole-v0`): 
```
python train.py
```

For rendering the learned behavior:
```
python render.py
```

The agent should be runnable on any environemnt with a discrete action space. To run the agent on some other environment, type `python train.py -env ENVIRONMENT_NAME`.

For example, the same architecture can also solve `Acrobot-v1`: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-Acrobot-v1.gif" width=400>


... and `LunarLander-v2`: 

<img src="https://github.com/qihongl/demo-advantage-actor-critic/blob/master/figs/render-LunarLander-v2.gif" width=400>



### dir structure: 
```
.
├── LICENSE
├── README.md
├── figs                            # figs           
├── log                             # pre-trained weights 
├── requirements.txt
└── src
    ├── models
    │   ├── _A2C_continuous.py      # gaussian A2C
    │   ├── _A2C_discrete.py        # multinomial A2C
    │   ├── _A2C_helper.py          # some helper funcs 
    │   ├── __init__.py
    │   └── utils.py                
    ├── render.py                   # render the trained policy 
    ├── train.py                    # train a model 
    └── utils.py

```

### Reference: 

[1] 
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

[2] 
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. Retrieved from http://arxiv.org/abs/1606.01540

[3] 
<a href="https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py">pytorch/examples/reinforcement_learning/actor_critic</a>

[4] 
<a href="http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf">Slides</a> from 
Deep Reinforcement Learning, CS294-112 at UC Berkeley
