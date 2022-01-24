# RL-CourseProject

## DQN
Reading and coding up the DQN paper to solve any of the discrete-space, classic control environments by OpenAI gym. Here *'CartPole-v0'* is used. Testing the proposed measures, namely the target network and the replay memory for a general case of function approximation using a feed-forward neural net and observing the learning curves (average reward over the last 100 episodes vs the number of episodes). Environment considered solved if and when the curve touches 195.

Reporting plots on performances, observations and inferences on hyperparameter variation. 

Analysis of performance and inferences drawn on removal of the target network and the transition replay buffer (as proposed by the original DQN paper for smooth and stable learning of the neural nets). Conclusion drawn on the relative importance of either, based on the breakdown of learning observed from the plots.  

*Reference*: DQN paper (https://arxiv.org/pdf/1312.5602)

## Gridworld
Creating a custom gridworld (with
puddles) of given shape, possible start states, reward signals and several
other conditions, using OpenAI gym API. 

![The puddle world](https://github.com/Anshu1245/RL-CourseProject/blob/master/Gridworld/the_puddle_world.jpeg)

**Environment details**
1. This is a typical grid world, with 4 stochastic actions. The actions might result in
movement in a direction other than the one intended with a probability of 0.1. For
example, if the selected action is N (north), it will transition to the cell one above your
current position with probability 0.9. It will transition to one of the other neighbouring
cells with probability 0.1/3.
2. Transitions that take you off the grid will not result in any change.
3. There is also a gentle Westerly blowing, that will push you one additional cell to the
east, regardless of the effect of the action you took, with a probability of 0.5. 
4. The episodes start in one the start states in the first column, with equal probability.
5. There are three variants of the problem, A, B, and C, in each of which the goal is in
the square marked with the respective alphabet. There is a reward of +10 on reaching
the goal.
6. There is a puddle in the middle of the gridworld which the agent would like to avoid.
Every transition into a puddle cell gives a negative reward depending on the depth of
the puddle at that point, as indicated in the figure.

Use them for training an
agent using *TD methods*, such as Q-Learning and SARSA, and report
performance plots, showing average rewards and average steps to goal per
episode, and optimal policies in each case.

It was also used for training
*Eligibility Trace methods* such as SARSA(𝝺) and report similar
performance plots for various values of 𝝺 in regular intervals in [0, 1].
