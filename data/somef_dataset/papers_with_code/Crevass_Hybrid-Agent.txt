# Hybrid-Agent
On/off-policy hybrid agent and algorithm with LSTM network and tensorflow.
A method of hybrid agent and training algorithm using both on-policy loss function and off-policy loss function, reference to DDPG(http://arxiv.org/abs/1509.02971) and DPPO(http://arxiv.org/abs/1707.06347).

Require tensorflow, openAI gym and mujoco to train the agent.

# Start Training
To start training a agent, run **testrun.py**. Tune the parameters in this file as you like.Either to train a new agent with **Restore_iter = None** or restore network weights with **Restore_iter**.
Tensorflow ckpt files will be saved in **tf_saver**, video of environments will be saved in **video**, and replay buffer's data will be saved in **replays**.
