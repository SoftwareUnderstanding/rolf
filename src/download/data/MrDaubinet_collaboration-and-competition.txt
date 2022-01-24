# Project 3: Collaboration and Competition
## Train two agents to play tennis against one another


![real world image](images/robots.gif )

# Background

In the field of reinforcement learning, new algorithms are tested in simulated game environments. These allow agents to learn at accelerated speeds, not possible in the real world. While some agents may not have real world aplicability, like the one in this project, others can be initially trained in a simulated environment and continue to learn in a real world environment. The purpose of this project is to train an agent to play tennis by competitivly learning against itself.  

![real world image](images/robot.gif )

# The Environment
This project is based off of the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) Environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.


The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Methodology

The Experimental setup is as follows:

  1. Setup the Environment.
  2. Establish a baseline with a random walk.
  3. Implement a reinforcement learning algorithm.
  4. Display Results.
  5. Ideas for future work.


## 1. Environment Setup

Download the environment from one of the links below. You need only select the environment that matches your operating system:
  * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Import the Unity environment and create an env object
```python
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='location of tennis.exe')
```

Info about the environment is printed out through the ```Info()``` class found [here](https://github.com/MrDaubinet/collaboration-and-competition/blob/master/info.py)  as seen below:
```
Unity Academy name: Academy
Number of Brains: 1
Number of External Brains : 1
Lesson number : 0
Reset Parameters :

Unity brain name: TennisBrain
Number of Visual Observations (per agent): 0
Vector Observation space type: continuous
Vector Observation space size (per agent): 8
Number of stacked Vector Observation: 3
Vector Action space type: continuous
Vector Action space size (per agent): 2
Vector Action descriptions: ,
created Info
Number of agents: 2
Number of actions: 2

States look like: 
[ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
States have length: 24
```

## 2. Establish a baseline

To evaluate the difficulty of the environment. A random walk was scored before any algorithmic implementation of the reinforcement learning agents was made. This was done by randomly selecting actions to interact with the evironment for a set number of steps. 
```python
def run(self, max_t=1000):
  brain_name = self.env.brain_names[0]
  brain = self.env.brains[brain_name]
  env_info = self.env.reset(train_mode=False)[brain_name]   # reset the environment    
  num_agents = len(env_info.agents)
  states = env_info.vector_observations                     # get the current state (for each agent)
  scores = np.zeros(num_agents)                             # initialize the score (for each agent)
  for t in range(max_t):
    actions = np.random.randn(num_agents, self.action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                       # all actions between -1 and 1
    env_info = self.env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations              # get next state (for each agent)
    rewards = env_info.rewards                              # get reward (for each agent)
    dones = env_info.local_done                             # see if episode finished
    scores += env_info.rewards                              # update the score (for each agent)
    states = next_states                                    # roll over states to next time step

  print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
  return scores
```

As you can see below, these guys are pretty retarded. 

![real world image](images/baseline.gif )


## 3. Implemented Algorithm
Similar to the reasoning pointed out in the previous [Continuous Control](https://github.com/MrDaubinet/Continuous-Control) Project. Due to the nature of the environment being a continuous control problem. The reinforcement learning agorithm needs to be able to work in a continuous space. This hard requirement means we have to use a deep learning approach where neural networks are used for continuous function approximation. When considering between Policy-based vs Value-based Methods. Policy-based methods are better suited for continuous action spaces. I selected the [https://arxiv.org/pdf/1706.02275.pdf](https://arxiv.org/pdf/1706.02275.pdf) algorithm, since I had already implemented the ddpg algorithm for the previous project and it could be adapted to the maddpg algorithm with some minor changes.

I copied the Actor and Critic models, as [found here](https://github.com/MrDaubinet/Continuous-Control/blob/master/model.py), but I removed batch normalization from the actor model and changed the critics input shape to accept states and actions from both agents. I copied the agent code, [found here](https://github.com/MrDaubinet/Continuous-Control/blob/master/agent.py), then changed it to accomidate a single environment, where both agents share the critic model.

The ```MADDPG()``` code can be [found here](https://github.com/MrDaubinet/collaboration-and-competition/blob/master/maddpg.py) resulting DDPG algorithm can be seen below:
```python
    def train(self, n_episodes=5000, max_t=int(1000)):
        scores_deque = deque(maxlen=100)
        scores = []
        average_scores_list = []

        for i_episode in range(1, n_episodes+1):                                    
            env_info = self.env.reset(train_mode=True)[self.brain_name]     
            states = env_info.vector_observations               
            score = np.zeros(self.num_agents)

            self.reset()

            for t in range(max_t):
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]            
                next_states = env_info.vector_observations
                rewards = env_info.rewards         
                dones = env_info.local_done                         
                self.step(states, actions, rewards, next_states, dones, t)        
                states = next_states
                score += rewards  

                if any(dones):                                 
                    break

            score_max = np.max(score)
            scores.append(score_max)
            scores_deque.append(score_max)
            average_score = np.mean(scores_deque)
            average_scores_list.append(average_score)

            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)), end="")  

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage score: {:.3f}'.format(i_episode , average_score))

            if average_score >= 0.5:
                self.save_weights()
                print("\rSolved in episode: {} \tAverage score: {:.3f}".format(i_episode , average_score))
                break
        return scores , average_scores_list

```

## 4. Results
The model was not very consistent in achieving the required score (0.5) and often when it did, it would proceed to decrease in score drastically if left to train more. In one one of the better runs, it was able to achieve the result in around 1000 epochs, in others it could take longer. Worst of all, in allot of approached, the average moving score would gradually increase and then suddenly drop and become terrible

```
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DDPG')
plt.plot(np.arange(len(scores)), avgs, c='r', label='moving avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left');
plt.show()
```
![Results](images/score.png)


To see one of them in action check the gif below.

![real world image](images/baseline.gif )

Not the smartest kids on the block, they met "minimum requirements", but they don't seem all that great. It could be that these guys trained for too long and become unstable, clearly room for improvement. 

## 5. Ideas for Future Work
* **Code Improvements** - I think the models could be made more robust, perhaps with a different approach to implementing MADDPG and how the shared buffer is used.  
* **Hyperparameter optimization** - Most algorithms can be tweeked to perform better for specific environments when by changeing the various hyper parameters. This could be investigated to find a more effective agent.
* **Priority Experience Replay** - Prioritized experience replay selects experiences based on a priority value that is correlated with the magnitude of error. This replaces the random selection of experiences with an approach that is more intelligent, as described in [this paper](https://arxiv.org/pdf/1511.05952.pdf). 

# Get Started
1. Install Anaconda from [here](https://www.anaconda.com/). 
2. Create a new evironment from the environment file in this repository with the command 
    ```
    conda env create -f environment.yml
    ```
3. Run ```python main.py```

    Remove the comments in main to train and run the baseline.

4. Watch the agents vs one another.