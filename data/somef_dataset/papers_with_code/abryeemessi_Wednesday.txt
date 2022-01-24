# Wednesday practicals

Deep reinforcement learning: Deep learning, meet reinforcement learning.

Core idea is to train a deep Q network on Breakout Atari game.

Note: We will use deterministic version of Breakout environment. Literature often reports performance on stochastic version (more realistic task).

## Running the code 

Only one file `atari_dqn.py` with different parameters. For full details command `python atari_dqn.py -h`

Examples:

* Train for 10 000 agent steps (calls to `env.step`): `python atari_dqn.py --steps 10000 path_where_to_store_model`
* Enjoy trained model: `python atari_dqn.py --show --evaluate --limit-fps path_to_trained_model`
* Evaluate trained model: `python atari_dqn.py --evaluate path_to_trained_model`
* Store console output to a file `log.txt`: `python atari_dqn.py --log log.txt path_where_to_store_model`
    * **Note: Once you start properly training agents, it is a good idea to store these logs for future reference**

## Tasks

1. Start by filling in required parts to run the code:
    * Implement the network in `build_models`
    * Implement target-network update in `update_target_model`
2. You can now study the environment with `--show` and `--limit-fps` arguments to show the game.
    * You can also print out rewards and actions after the `env.step(action)` for better view of the game.
    * When does agent reward?
    * What is one episode? 
3. Try training the agent with `python atari_dqn.py --steps 3000 dummy_model`
    * What do the results look like? Good? Bad?
    * What do you think is the problem (what was also problem with default Q-learning)?
4. Implement epsilon-greedy exploration in `get_action`
    * With probability EPSILON, take random action instead of the greedy action (already implemented in `get_action`)
    * You can use fixed EPSILON. A small probability should do the trick (e.g. 10% or 5%)
    * Try training again with `python atari_dqn.py --steps 3000 dummy_model`
    * What is different compared to previous run?
5. We are printing out very limited info. At the very least we should print out the loss of training. 
    * Implement printing average loss
        * In supervised learning (recall Monday's imitation learning), loss tells how accurate the network is.
        * This is not quite as straight-forward with Deep Q learning, **but** it still is a vital debugging tool to see
          if something is wrong with training the network
        * Skip to the end of `main()` function where you can find our print messages. Include average loss 
          of the episode in the print-out message.
    * Train agent with `python atari_dqn.py --steps 2000 dummy_model`
    * What does the loss look like? Does it decrease/increase? It should decrease in the longer run, and it should not explode (be high values above 1000)
    * Something is wonky. Take a look at `update_model` and see if you can fix the problem with loss.
    * Run agent with `python atari_dqn.py --steps 2000 dummy_model` again and see if loss seems more reasonable now
6. It is still hard to say if agent is improving to right direction or not. Implement more monitoring.
    * Implement tracking average reward
        * Class `collections.deque` creates a list with maximum size. If maximum size is reached, oldest element is dropped.
        * Create a `deque` that stores episodic rewards from last 100 episodes (or less)
        * After each episode, print out the average reward from last 100 episodes
    * We know what Q-values should be like (not negative, well above zero), so we can track that as well.
        * We can monitor if we are even going to right direction by printing average Q-value per episode. Implement this.
        * For every episode, store the sum of all predicted Q-values and number of them (used to calculate average).
        * Note that `get_action` function returns Q-values and the selected action.
        * Print average Q-value after every episode. 
    * Try training again with `python atari_dqn.py --steps 10000 dummy_model`
    * Does the monitoring tell you anything useful? Average episodic reward might not, but what about average Q-value? 
7. Try "optimistic exploration" again by initializing Q-values to something high.
    * Not as trivial as setting all values in a table to specific value, since we work on networks.
    * A simple and crude way to do this: Initialize weights (kernel) of the final layer (output layer) to zero and biases to one.
        * End result: Before updates, the Q-network will predict one for all states.
        * See documentation for [`Dense` layers](https://keras.io/layers/core/) for how to change initial values.
8. Try training agent for a longer time with `python atari_dqn.py --steps 50000 proper_model`
    * How high average reward did you get? 
    * Evaluate and enjoy the model after training. What are the subjective / objective results?
        * You can enjoy your agent with `python atari_dqn.py --evaluate --show --limit-fps proper_model`
        * You can evaluate your agent with `python atari_dqn.py --evaluate proper_model`
9. Try reaching higher average reward by tuning the exploration and other parameters.
    * With some tinkering, you should be able to get to reliable 4.0 average reward in under 100k training steps.

Extra things to try out:

* Visualize Q-values while enjoying using Matplotlib and interactive plots.
* Try the code on `BreakoutNoFrameskip-v4` environment (set with `--env` argument).
    * Same as `Breakout-v0`, but with "sticky actions": With some probability, the next action is 
      equal to previous action rather than the one given in `env.step(action)`.
* Try the DQN code on different Atari game (e.g. Pong).

Extra implementing:

* Implement Double DQN (modification to `update_model`): http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/doubledqn.pdf
* Implement Dueling DQN (modification to `build_models`): https://arxiv.org/pdf/1511.06581.pdf
* Note that you can implement both together
