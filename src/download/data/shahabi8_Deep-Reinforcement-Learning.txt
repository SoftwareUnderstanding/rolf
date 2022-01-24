# Deep-Reinforcement-Learning
## Train a Quadcopter How to Fly
The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own.
## Instruction
* task.py: Define task (environment) for reinforcement learning.
* physics_sim.py: This file contains the simulator for the quadcopter.
* Replay_buffer.py: Most modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples.
* Actor.py DDPG: Actor (Value) Model.
* Critic.py DDPG: Critic (Value) Model.
* DDPG.py DDPG agent to put together actor and critic.
* OUnoise.py Ornstein–Uhlenbeck Noise. Use this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly. 
## Libaries Used
* Keras
* csv
## Additional Document
Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning. (https://arxiv.org/pdf/1509.02971.pdf)
