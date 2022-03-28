# Baxter-VREP
## TD3 Algorithm
The TD3 algorithm uses two critic networks and selects the smallest value for the target network.  To prevent overestimation of policies propogating errorthe policy network is updated after a set number of timesteps and the value network is updated after each time step. Variance will be lower in policy network leading to more stable and efficient training and ultimately a better quality policy.  For this implementation, the actor network is updated every 2 timesteps.  The policy is smoothed by adding random noise and averaging over mini-batches to reduce the variance caused by overfitting. <br/>
#

![TD3 Algorithm from https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93](/images/TD3-algorithm.png)
### Project assumptions
## Papers Consulted
1.  [Van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with Double Q-Learning. In Thirtieth AAAI conference on artificial intelligence(2016).](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)<br/>
In order to reduce bias this method estimates the current Q value by using a separate target value function. 
2. [Hasselt, H. V. Double Q-Learning. In Advances in Neural Information ProcessingSystems(2010), 2613–2621.](https://pdfs.semanticscholar.org/644a/079073969a92674f69483c4a85679d066545.pdf?_ga=2.3333939.269109633.1566140607-1006436993.1566140607)<br/> 
In actor-critic networks the policy is updated very slowly making bias a concern.  The older version of Double Q Learning uses clipped double Q learning.  This takes the smaller value of the two critic networks (the better choice).   Even though this promotes underestimation, this is not a concern because the small values will not propogate through the whole algorithm.
3. [Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)<br/>
Original citation for the PyTorch implementation fo Twin Delayed Deep Deterministic Policy Gradients (TD3), [source code](https://github.com/sfujim/TD3)
4. [Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Prioritized experience replay.arXiv preprint arXiv:1511.05952(2015).](https://arxiv.org/abs/1511.05952)<br/>
Prioritized experience replay- See Overleaf article summary.

## Code References 
1. [TD3 Algorithm Code](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93) from Towards Data Science implementation of Addressing function approximation error in actor-critic methods.
2. [OpenAI Gym, Replay Buffer and Priority Replay Buffer](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) 
3. [TD3 Implementation](https://arxiv.org/abs/1802.09477) Used for TD3 algorithm implementation.
4. [DQN code from Richard Lenz, UNF]()


## Bellman Equation Notes
* State: what the agent is observing at a time step
* Action: the input the agent provides to the the environment, calculated by applying a policy to the state
* Reward: the feedback for the action

## Links
* [Getting started with Vrep](http://hades.mech.northwestern.edu/index.php/Getting_Started_with_the_V-REP_Simulator)
* [CoppeliaSim User Manual](http://www.coppeliarobotics.com/helpFiles/index.html)
* [Vrep/Python instructions](http://fid.cl/courses/ai-robotics/vrep-tut/pythonBubbleRob.pdf)
* [ROS Robotics by Example](https://dl.acm.org/citation.cfm?id=3200107) Baxter reference for ROS including: joint angles,... (download the book)[https://drive.google.com/open?id=11UpOH1fZd1qhXr9i8tEyVa1g4NVmL-me]

### Exporting Virtual Environment Packages
Export a list of packages
```
pip freeze > requirements.txt
```

Install packages
```
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```
### Run VREP Headless
Launch VREP with following command. (You'll need to update the path to the vrep file.)
      
    V-REP/vrep.sh -h -q /home/cislocal/Jupyter/V-REP_Scenes/baxter.ttt -gREMOTEAPISERVERSERVICE_19999_FALSE_FALSE
Then in the Vrep_SIM class I had the following line to start the simulation:
           
    errorCode = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
            
I placed this line after the print('Connected to remote API server') in the Vrep_SIM class.
