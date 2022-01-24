# acrobotDDQN
My first Double DQN model. Used to learn a basic control task -- Acrobot -- [OpenAI Gym](https://gym.openai.com/envs/Acrobot-v1/)

see my DQN for [Cartpole](https://github.com/JustinStitt/cartpoleDQN) and [Lunar-Lander](https://github.com/JustinStitt/lunarLanderDQN)
for a much more detailed writeup on DQN's.

or see my [Digit Recognizer](https://github.com/JustinStitt/digitRecognizer) for a supervised Deep Learning model.

For Acrobot, I've implemented a [Double Deep Q-Network](#Double-DQN). A DDQN uses two networks, one local network and one target network.
Before I get into that, let's look at the results.


**Untrained Agent**

![](visuals/untrainedGIF.gif)

**Agent after 1000 epochs**

![](visuals/trainedGIF.gif)

The trained agent may look sporadic but it is just achieving its objective as fast as possible.

**Acrobot's Objective**

The objective of Acrobot is to cross the black threshold line near the top of the window. Acrobot can control two joints by 
adding torque to the left or to the right. Crossing the threshold faster is better, this is why you will see the trained agent move
extremely quickly side-to-side to generate momentum.

# DQN vs DDQN

### Standard DQN

A standard Deep Q-Network agent has just one network. This network is what is being trained after each iteration. 


### Double DQN

A Double Deep Q-Network agent, true to its name, has two networks. One local network that is trained every iteration just like a 
standard DQN, but it also has a target network that is used to predict the best action that could've been made in any given state.

**The Problem With a DQN and How a DDQN Solves It**

The issue with a DQN is that we are using the same network to both train and calculate the best action for a given state.

For insight, here is the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation):

![](visuals/bellman_eq.png)

Here, the same Q evaluation network is being used for both the current (state,action) and (next_state, next_action) -- seen underlined in blue-- . This yields some
issues because you are actively training the same network that you are using to evaluate actions to ultimately calculate loss.
Without a target network, the local network is almost blindly following an ever-changing network.

A DDQN solves this issue because it separates the duality of training and evaluating.  The introduction of a target network adds some stability to the training process
which allows, generally, for better training results.

The target network is updated via a HYPERPARAMETER which counts iterations passed before the last update. The update process 
involves creating a soft copy of the local network and assigning it to the target network.

## References and Useful Links:

https://towardsdatascience.com/double-deep-q-networks-905dd8325412

https://arxiv.org/abs/1509.06461

[Phil Tabor's excellent repository on all things DQN](https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code)