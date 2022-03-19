[//]: # (Image References)

[image1]: train_e5.gif "Untrained Agent"
[image2]: train_e15.gif "In process of training"
[image3]: trained.gif "Trained Agent"
[reward]: reward.png "Plot of reward"

# TD3 Agent implementation. Solution for Reacher enviroment in Unity ML.


## TL;DR;

Repo with Implemenation of TD3 algo for Reacher 20 env. from Udacity course. First reaches reward 30 in 21 episode, gets a \[0,100\] mean of 30 in 100.  

Proposed enviroment has 20 Reachers simultaneously. As this is a sync. process, I choosed a TD3 algo that collects expirience 20x faster with 20 hands, treating them as sync. process. I see no sence in using async. approach here, as enviroment isn't truly async.

#### Trained agent looks like this
![Trained Agent][image3]

## Description of files
 - Demo.ipynb - allows you to check enviroment and see working agent example
 - Solver.ipynb - reproduces the training procedure
 - agent.py - TD3 agent implementation
 - networks.py - actor and critic Pytorch definitions
 - replay_byffer.py - Replay Buffer implementation from OpenAI Baselines 
 - actor.pth - Saved weights for Actor network from TD3
 - critic.pth - Saved weights from Critic networks from TD3
 
## About the problem

Reacher20 is a Udacity version of Reacher enviroment, containing 20 simultaneous Reacher agents.
Each agent get a state vector of 33 floats, describing it's joints positions and speed as well as ball's position and speed.
So in this case state is a (20, 33) vector.

At every time step you have to perform an action for every agent, where action is a torque applied to the joint. Action is defined by floats in range (-1,1) and every agent needs 4 actions. So for 20 agent action space is (20,4).

Agent get rewarded if its end is localized in a moving spherical space. Game is considerend solved when mean reward from 20 agents in the last 100 episodes is >= + 30.


## About the algo and results
I've initially tried DDPG and A2C. While I had some progress and reward was growing, training process was quite unstable. As I was still in the exploratory phase, I switched to TD3.
It worked like a charm from the first attempt. Why search for anything else?  https://arxiv.org/pdf/1802.09477.pdf

### Neural networks details

1. Actor a.k.a. policy network.
Simple Fully connected network.  
FC: (33) -> (400) -> (300) -> (4) with ReLu actiovations.  
Imporant to note, last operation in Tanh activation, that scales action output to required env. spec (-1,1).  
You can do sigmoid and rescale if you prefer.

Used to predict optimal action.
```python
x = self.max_action * torch.tanh(self.l3(x)) 
```

2. Critic a.k.a. value network.
Again 2 critic networks are implemented inside one pytorch Module. 
Both are symmetrical FC networks.
FC: (33 + 4) -> (400) -> (300) ->(1) with ReLu activations.
Used to predict reward for a certain action made in a certain state.

### Hyper-parameters.

policy_freq = 2 # Policy network is updated every Xnd step  
batch_size = 512 # N of samples sampled from buffer to step of training  
discount = 0.99 # reward discount  
replay_buffer = int(1e5) # size of replay buffer  
policy_noise = 0.1 # amount of noise added to predicted action. 
tau = 5e-3 # How aggresive target network is updated at each update step (1 - fully update, 0 - no update)  

![Reward fucntion vs episode][reward]

## Getting started
1. Download the environment from one of the links below.

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Install Python requirements

2.1. Use my env
Attn! This can be quite heavy with some packages not needed for this project. 
```bash
pip install -r requirements.txt
```

2.2 Use your own
It's likely that you will need only standard package of numpy, pytorch to make this work.


## Things to improve
1. TD3 is quite stable. I choosed a random stack of hyperparams and they worked for 20 and for 1 Reacher agent. Some tuning maeks sence
2. Why not trying Prioritized Expririence Replay here?


## Samples of enviroment and training process.

#### Untrained agent
![Untrained Agent][image1]

#### Agent at episode 15 (half way trained)
![Episode 15][image2]



## Acknowledgements

1. Replay Buffer module was taken from Baseline package from Open AI [https://github.com/openai/baselines]
2. TD3 is based on Medium explanaition [https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93]
