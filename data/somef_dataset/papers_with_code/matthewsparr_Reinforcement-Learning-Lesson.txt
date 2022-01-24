
# Reinforcement Learning

## Introduction

You may have heard that the two types of learning are supervised and unsupervised - either you are training a model to correctly assign labels or training a model to group similar items together. There is, however, a third breed of learning: reinforcement learning. Reinforcement learning seeks to train a model make a sequence of decisions.

## Objectives

You will be able to:

* Explain what reinforcement learning is and its uses
* Implement reinforcement learning in your own projects

## What is reinforcement learning?

Reinforcement is training a model to achieve a certain goal through the process of trial and error. In RL, there are 5 main components: the <b>environment</b>, the <b>agent</b>, <b>actions</b>, <b>rewards</b>, and <b>states</b>. Those five components interact in the way shown in the diagram below.

<img src = "rl_chart.png"></img>
<br>
<br>
The agent is simply the model or artifical intelligence being trained. The environment is the interactive world in which the agent acts. The states are different, distinct versions of the environment. Actions are performed by the agent on the environment. The rewards are positive or negative feedback values corresponding to the actions taken and the effect they have on the environment.

Consider this:
<br>
<br>
<img src = "penguin_game.png"></img>
<br>
<br>
This image represents a possible environment. In this environment, the penguin in the bottom left corner would be the <b>agent</b>. It ultimately wants to end up at the fish in the upper right corner. However, there are obstacles in its way there - the shark and the mines. The penguin must make a series of <b>actions</b> that will allow it to reach the fish while avoiding the obstacles. The <b>rewards</b> in this case could be as follows:<br> <br>
<center>+1 if the penguin moves to an empty ocean space<br>
-1 if the penguin runs into the "wall" of the environment<br>
-50 if the penguin moves to the shark space<br>
-100 if the penguin moves to a mine space<br>
+100 if the penguin moves to the fish space</center>
<br><br>
So as the penguin chooses an <b>action</b> each turn, it is then presented with a new <b>state</b> and given a <b>reward</b> depending on what type of space it moved onto. With this new information, the penguin or <b>agent</b> will then evaluate the new <b>state</b> and select the next<b>action</b>.
<br>
<br>
Over time, through enough trial and error (and the penguin hopefully narrowly escaping death by shark or mine), the penguin would ultimately learn the best route to the highest <b>reward</b> - navigating to the fish while avoiding the mines and the shark.

This is a very simple environment but it demonstrates the fundamentals of reinforcement learning.

## Why reinforcement learning?


So you understand the basics of RL and how it applies to our penguin's world above, but when does it make sense to use this type of learning in the real world?

### Robotics
Teaching robots to perform various tasks.

### Chemistry
Optimizing chemical reactions.

### Traffic light control
Optimizing traffic flow.

### Resources management in computer clusters
Allocation of limited computational resources.

### Finance
Portfolio optimization, risk management, stock trading.

### Video games
AIs capable of beating human performance (Deepmind, AlphaGo, AlphaZero), improving computer-controlled players

And so many more! The field of reinforcement learning will continue to grow and can be applicable to any field where there is a need to improve decision making.

## How is reinforcement learning implemented?
Now for the exciting part - learning how to implement RL!


### RL Algorithms
There are many different reinforcement learning algorithms. These include <b>Q-Learning</b>, <b>State-Action-Reward-State-Action (SARSA)</b>, <b>Deep Q Network (DQN)</b>, and <b>Deep Deterministic Policy Gradient (DDPG)</b>, as well a myriad of different iterations of each.
<br>
<br>
Each of these has its own merits and use cases. By far, the most commonly used, however, would be <b>Q-Learning</b> and its neural network based extension, <b>Deep Q Networks</b>, so we will start with those two.

### Q-Learning
Q-Learning seeks to maximize what's known as the <b>Q-value</b> in order to select the most optimal action at any given state. <br>
You can think of 'Q' as in <i>quality</i> as the value represents the "goodness" of an action.
<br>
<br>
As an agent makes actions, it updates the Q-values in the <b>Q-table</b>. The Q-table consists of all possible combinations of states and actions. So for the penguin example above, the environment is 5x5 which  means there are 25 possible states in which the penguin could be. At each of those states, the penguin has 4 possible moves - up, down, left, or right. Together, that gives a total Q-table size of 100.
<br>
<br>
Now for the Q-values, they are calculated with the formula below.

<img src = "q_formula.png"></img>


<b>Q(st,at)</b> represents the Q-value at a given state, s(t), and given action, a(t).<br><br>
<b>r(t)</b> represents the reward<br><br>
<b>γ</b> (or gamma) is a discount factor<br><br>
<b>maxQ(s(t+1),a(t+1))</b> is the maximum Q-value at the next state (s(t+1)) which is found by testing all possible actions

What this function does is allow the agent to learn to select not just the action that will give the highest immediate reward, but it will take into account future rewards as well.

The gamma value is necessary because it helps to balance immediate v.s. future reward. We would like the agent to value immediate rewards slightly more because they are more guaranteed than the predicted rewards of the next state.

The main issue with Q-Learning is that, while it is great at being able to select actions for states it has seen before, it is not able to predict anything in new states that it hasn't seen before.
<br>
<br>
This is where <b>Deep Q-Learning</b> comes into play.

### Deep Q Network (DQN)
DQNs take the basic principles of Q-Learning and improves upon them by estimating Q-values through the use of a neural network. DQNs, unlike regular Q-Learning, are able to handle never-before-seen states.
<br>
<br>
In a DQN, the input is usually the current state while the output is the Q-values for each of the possible actions.
<br>
<br>
There are three essential components that allow a DQN to work.

### Epsilon Greedy Policy
In order to gain experiences properly when starting off, a DQN employs a policy in which a value, epsilon, is used to tell the network whether to take a random action or predict an action. This value usually starts off at 1 and decays over time according to some decay rate (usually a decimal close to 1, i.e. 0.99).
<br><br>
What this means is that at the beginning of training, with epsilon at 1, the network will always perform random actions. Since it hasn't really been trained yet, this is the best strategy to <b>explore</b> the environment.
<br><br>
Then, as epsilon decays below 1, there is a growing chance each time of predicting an action instead of choosing randomly.
<br><br>
As the network is trained more and more, it will start to predict actions more as well.
<br><br>
Eventually, epsilon will approach zero, the model will be fully trained, and all actions will be predicted.

### Experience Replay
When humans learn something by trial-and-error, we don't just look at our most recent attempt and base our next decision solely off of that. Instead, we rely on our memory of all our past attempts. DQNs must do something similar.
<br><br>
Experience replay means that when the network is trained, it is not trained on each action it takes, as it takes them. Instead, a history of all states, actions, and corresponding rewards are stored in a memory. Then, at given intervals, the network is trained on a random sample of that memory according to some batch size.
<br><br>
This helps to decouple the temporal relationship between subsequent turns and greatly improves the stability of training.

### Seperate Target Network
Normally, when training the network, we have to perform predictions using the same network to calculate the Q-value updates. This causes an issue where, as we are training the network, the Q-values are constantly shifting. This can create a feedback loop where the 
<br><br>
What this means is that at the beginning of training, with epsilon at 1, the network will always perform random actions. Since it hasn't really been trained yet, this is the best strategy to <b>explore</b> the environment.
<br><br>
Then, as epsilon decays below 1, there is a growing chance each time of predicting an action instead of choosing randomly.
<br><br>
As the network is trained more and more, it will start to predict actions more as well.
<br><br>
Eventually, epsilon will approach zero, the model will be fully trained, and all actions will be predicted.

Here is a basic rundown of how training happens in a DQN.
<br>
<br>

#### <center>1) Intialize parameters and model </center>
#### <center>2) Choose action via epsilon-greedy policy</center>
#### <center>3) Store state, action, reward, and next state in memory</center>
#### <center>4) After n actions, train on random sample of batch size in experience replay</center>
#### <center>5) Transfer weights from primary model to target model</center>
#### <center>6) Decay epsilon value by decay factor</center>
#### <center>7) Repeat steps 2-6 until desired performance is reached</center>
<center> </center>

## Reinforcement Learning in action

Now for the exciting part - learning how to use RL!

The easiest way to quickly start experimenting with RL is to train models to play video games.
<br>
<br>
Fortunately, in Python there is a great library that enables this - Gym by OpenAi.

## Gym
<img src="gym_games.jpg"></img><br><br>
Gym is a library designed specifically to provide numerous game environments for training a reinforcement learning model.

https://gym.openai.com/

Gym can be install using pip:

<b>pip install gym</b> <br>
<br>
Windows users, if you want to run Atari environments, you will have to also install this: <br><br>
<b>pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py</b>

### Environments

Environments are the games provide by the library. These include simple text based games, 2D and 3D games, and even classic Atari games.

https://gym.openai.com/envs/

To get started, all you have to do is select an environment and then plug it into the code below. This will play out a number of games and moves, taking random actions at each step. (This code specifically using MsPacman but feel free to try any of the other games.)


```python
import gym
import time
env = gym.make('MsPacman-v4')
for game in range(1):
    state = env.reset()
    for t in range(500):
        time.sleep(0.05)
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            break
env.close()
```

Calling <b>reset()</b> on the environment will do just that - reset it to a new game. <br> <br>
The <b>render()</b> command updates the video output. <br><br>
The <b>step()</b> command causes the game to play out one step or move, which in this case is a random action given by the action_space.sample() line. This function returns the next state of the game, the reward for the given move, a boolean value 'done' telling us whether or not the game ended, and any additional info the specific game might provide.<br><br>
Check if <b>done</b> is true each time lets us break out of the current game and start a new one. <br><br>
Calling <b>close()</b> shuts the environment down, stopping the video render.

<b>Note:</b> playing the game in this way, moves happen very quickly. To slow down the render to a more human-level speed, you can <b>import time</b> and run <b>time.sleep(0.1)</b> to add a brief pause after each frame.

There is a lot to tweak within a Gym environment. You can modify different parameters such as the maximum number of moves in a game, the number of lives, and even the physics of some games. You can also create your own custom Gym environments. <br><br>
For example, this Medium article - https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e - shows how to create a custom Gym environment to simulate stock trading.

## Summary

That concludes this brief introduction into the wonderful world of Reinforcement Learning. Hopefully this provides you with a good foundation of understanding of the topic and can help springboard you into starting RL projects of your own.
<br><br>
Addtionally, it is important to note that RL is extremely expensive computationally. To train the Atari games, for example, even with a high end GPU, it could take multiple days of constant training. That being said, more simple environments can be trained is as little as 30 minutes to 1 hour. 
<br> <br>
Also important to note is that RL models are highly unstable and unpredictable. Even using the same seed for random number generation, the results of training may vary wildly from one training session to another. While it may have only taken 500 games to train the first time, the second time it might take 1000 games to reach the same performance.
<br><br>
There are also many hyperparameters to adjust, from the epsilon values to the layers of the neural network to the size of the batch in experience replay. Finding the right balance of these variables can be time-consuming, adding to the time sink that RL has the possibility to become.
<br><br>
That being said, it is extremely fulfilling when you see that your AI has actually started to learn and you see scores start climbing. <br><br>
Beware, you may just find yourself become emotional involved in the success of your model.

## Additional Resources

http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf - paper on Q-learning<br>
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf - paper on DQN<br>
https://medium.com/machine-learning-for-humans/reinforcement-learning-6eacf258b265 - blog post on RL<br>
https://arxiv.org/pdf/1509.02971.pdf - paper on DDPG<br>


```python

```
