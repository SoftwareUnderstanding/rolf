# Carla the Reinforcement Self-Driving-Car (Version Morra).
<p>In this project I aim to create a self-driving car that uses a reinforcement learning approach to navigate in an open-source simulator for autonomous driving research called Carla. The data that the car will use as to guide its decision is image data in the `RGB` format and `collision` data. For more information about carla please visit the following Links:</p>

- https://carla.org/

- https://github.com/carla-simulator/carla

![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_desktop0.PNG?raw=true)

# Motivation.
<p> Created this project as part of my Master's Final project for the Year 2020 and a passion for reinforcement learning. </p>
This video from openAI really inspired me: https://www.youtube.com/watch?v=kopoLzvh5jY

### How to use this repo.
- Download anaconda
- Create a virtual environment: conda create -n envname python=3.7 anaconda
- pip install requirements.txt
- Download the Carla Repo from https://github.com/carla-simulator/carla
- Once everything is setup you must ensure that you have that you have CarlaUE4.exe running. or if you are on linux run the command ./CarlaUE4.sh

# What is in this repo
- Code for a reinforcement learning self-driving car.
- A step by step code breakdown in the form of a jupyter notebook.
- A modularized version of the code.
- Powerpoint presentation.
- Data and Graphs.
- A Readme with detailed instructions.
- Documentation.

### Carla environment. 

![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/carla_look1.PNG?raw=true)

This is how carla looks like from the inside. It is an extremely beautiful environment.

### Reinforcement Learning.
The main idea in RL is that you have an agent which is an "intelligent being" the interacts with an environment by means of taking actions and then receives feedback from the environment to indicate whether the agent has done well or bad. Like raising a  child , if he does well in school you encourage(REWARD) him if he doesn’t then you perhaps ground him (Penalize). and your child starts to adjust his behavior accordingly.

Note that a +ve reward indicates a reward and a -ve reward indicates penalty.

![preview](https://github.com/OMS1996/Carla_The_RL_Self-Driving-Car/blob/main/Images/rl_env1.PNG)

# DQN.
How the DQN algorithm generally looks like is as follows: courtesy of @deeplizard's website: https://deeplizard.com/learn/video/0bt0SjbS3xc
<!-- BLOG-POST-LIST:START -->
<!-- BLOG-POST-LIST:END -->
```
1.Initialize replay memory capacity.
2.Initialize the network with random weights.
3.For each episode:
  1.Initialize the starting state.
  2.For each time step:
    1.Select an action.
      Via exploration or exploitation
    2.Execute selected action in an emulator.
    3.Observe reward and next state.
    4.Store experience in replay memory.
    5.Sample random batch from replay memory.
    6.Preprocess states from batch.
    7.Pass batch of preprocessed states to policy network.
    8.Calculate loss between output Q-values and target Q-values.
      Requires a second pass to the network for the next state
    9.Gradient descent updates weights in the policy network to minimize loss."
```


### Demo video ( First few episodes ).
Here is the first few minutes of the training process for self driving car.
Please see the [video](https://www.youtube.com/watch?v=oAbDeb887_U) by [@OMS1996](https://github.com/OMS1996).
As you can see at the beginning it is not very smart but slowly but surely it begins to get smarter and smarter.

<details>
  <summary>Results!</summary>

  ![advanced](https://.png)
</details>

#### Potential improvements.
- [ ] Incoporate dynamic weather for a wider range of data. ( Level: Easy)
- [ ] Implement prioritized experience replay ( Level: Medium) https://arxiv.org/abs/1511.05952
- [ ] Create a perception system (Level: Hard)
- [ ] Attempt an improved DDQN (https://arxiv.org/abs/1509.06461)
- [ ] Dueling Network https://arxiv.org/abs/1511.06581
- [ ] Implement PPO 
- [ ] Implement A3C
- [ ] Create a model based self-driving car (Level: Hard)
- [ ] Combine RL + Rule based machine learning for self-driving car (level: Very hard)
- [ ] Use imitation learning

### Bugs.
If you are experiencing any bugs, please email me at omarmoh.said@yahoo.com

### Resources.
Following are some the sources that I used some are more important than others, but here they are:

| Name | Comments
|--------|--------
| [Udacity Deep reinforcement learning](https://www.udacity.com/course/reinforcement-learning--ud600) | `https://github.com/udacity/deep-reinforcement-learning` 
| [Human-level control through deep reinforcement learning](http://files.davidqiu.com//research/nature14236.pdf) | `http://files.davidqiu.com//research/nature14236.pdf` | http://files.davidqiu.com//research/nature14236.pdf
| [Issues in Using Function Approximation for Reinforcement Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf) | `http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf` 
| [Deep Learning Illustrated](https://www.amazon.com/Deep-Learning-Illustrated-Intelligence-Addison-Wesley/dp/0135116694) | `https://www.amazon.com/Deep-Learning-Illustrated-Intelligence-Addison-Wesley/dp/0135116694` 
| [Introduction to Reinforcement learning An Introduction](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.7692) | `http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.7692` 
| [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents) | `https://github.com/Unity-Technologies/ml-agents` (https://meta.stackexchange.com/questions/98771/what-is-my-user-id/111130#111130) and sub-domain
| [Grokking deep Reinforcement learning](https://www.manning.com/books/grokking-deep-reinforcement-learning) | `https://www.manning.com/books/grokking-deep-reinforcement-learning` 
| [Python Hands On machine learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291) | `https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291` 
| [David Silver Course Lecture 1](https://www.youtube.com/watch?v=2pWv7GOvuf0) | `https://www.youtube.com/watch?v=2pWv7GOvuf0` | Replace `playlistId` with your own Youtube playlist id 
| [Stanford Course Lecture 1](https://www.youtube.com) |  `https://www.youtube.com/feeds/videos.xml?channel_id=channelId` | Replace `channelId` with your own Youtube channel id 
instructions 
| [Helpful Github Repo](https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN) | `https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN` 
| [sentdex](https://www.youtube.com/user/sentdex) | `https://www.youtube.com/user/sentdex` 
| [MIT course lecture 1](https://anchor.fm/) | `https://anchor.fm/s/podcastId/podcast/rss` (https://help.anchor.fm/hc/en-us/articles/360027712351-Locating-your-Anchor-RSS-feed) 
| [Helpful Github Repo](https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN) | `https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN` 
| [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) | `https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0` 

### Thanks 
- Director of my program Professor.Andy Catlin
- Dean Paul Russo
- My supervisor Dr. Wonjun


### My Github account
* [My own GitHub profile readme](https://github.com/OMS1996) 

### How to contribute?
Make a pullrequest.



