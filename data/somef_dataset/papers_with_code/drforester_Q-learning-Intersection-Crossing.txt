# Q-learning Collision Avoidance  
Using a **_Deep Q Network_** to learn a policy to cross a busy intersection

![Alt text](images/2-lanes.gif?raw=true "Output") ![Alt text](images/4-lanes.gif?raw=true "Output")

Reinforcement learning (RL) methods have been successfully employed to learn rules, called policies, to solve tasks such as navigate a maze, play poker, auto-pilot an RC helicopter, and even play video games better than humans<sup>(1)</sup>. This last feat was accomplished in 2013 by DeepMind who were quickly bought by Google. Their paper “Playing Atari with Deep Reinforcement Learning”, describes how a computer can learn to play video games by taking the screen pixels and associated user actions (such as “move left”, “move right”, “fire laser”, etc.) as input and and receiving a reward when the game score increased.
Deep reinforcement learning combines a Markovian decision process with a deep neural network to learn a policy. To better understand Deep RL, and particularly the theory of Q-learning and its uses with deep neural networks, I read Tambet Matiisen’s brilliant Guest Post on the Nervana Systems website<sup>(2)</sup>. I say brilliant because even I could begin to understand the beauty and the elegance of the method. In looking for a Python implementation of this method that I could study, I found Eder Santana’s post “Keras plays catch”<sup>(3)</sup>, which describes his solution to the task of learning a policy to catch a falling piece of fruit (represented by a single pixel) with a basket (represented by three adjacent pixels) that can be moved either left or right to get underneath. In his post, he provides a link to the code. It was that post and code which most inspired this project.  

1. Playing Atari with Deep Reinforcement Learning, Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al., https://arxiv.org/abs/1312.5602  
2. https://www.nervanasys.com/demystifying-deep-reinforcement-learning  
3. http://edersantana.github.io/articles/keras_rl  

## Problem Statement
This project’s goal is to use a reinforcement learning method called Q-learning to arrive at a policy that an autonomous driving agent can use to cross a simulated busy intersection without incident and in the minimum time possible. That is to say that our agent should always wait until a large enough gap in traffic appears before driving through it, and it should never miss such an opportunity when presented. Our agent should be neither too timid nor too brave.  

The environment world is 21 pixels high by 21 pixels wide. All cars, including the agent, are of length 3 pixels. At each time step, each car moves forward one pixel; the agent upwards, and the traffic leftwards.

In the environment, the agent is not required to stop if there happens to exist a gap in the traffic large enough, otherwise it must come to a stop until such a gap presents itself (that’s this agent’s *raison d’être*). Since vehicles advance by one pixel per time step, we can say that, in the case of a single lane, a gap in traffic must be at least 3 pixels wide in order for the agent to “squeeze through”.  

The environment is instantiated in the following manner:  
env = Drive(grid_dims)  
where grid_dims is a tuple such as (21,21) representing the pixel height and width of the simulated world. Next the environment is set to initial conditions with:
env.reset()  
Traffic is placed on the road with randomly chosen gap lengths. Gaps must be at least one pixel wide. In practice they are usually between one and four, though they can sometimes be several pixels wider. The following call moves the traffic leftward one pixel:
env.propagate_horz()  
The agent then determines if it is at the intersection with the call:
env.at_intersection()
If it is not at the intersection, then the only valid action is “forward”. Otherwise, the agent must decide whether to go forward or remain. When the action is determined, the agent action is implemented by:
env.propagate_vert(action)
The action input to this method is what we must learn.

The Experience Replay functionality, and the reasons for using it is explained in reference 2 in this way:
“When training the network, random minibatches from the replay memory are used instead of the most recent transition. This breaks the similarity of subsequent training samples, which otherwise might drive the network into a local minimum. Also experience replay makes the training task more similar to usual supervised learning, which simplifies debugging and testing the algorithm.”
The experience replay implementation proposed here is almost identical to that Python class used in reference 3. Only one minor change is necessary to make it work with the output of the network described above.  
The following software have been used:  
* Python 3.6  
* Numpy  
* Scipy (namely: scipy.ndimage.interpolation.shift)  
* Matplotlib  
* Keras  
* TensorFlow  

## Evaluation Metrics
To evaluate the learned model and the Q-value implementation, a python program was written which instantiates an environment and loads the trained model. This testing program presents the agent with thousands of randomly generated environments and records the following two evaluation metrics:
1. Percentage of successful crossings
2. Number of missed crossing opportunities

## Project Design
The project consists of three Python programs, one for training, one for testing, and a plotting utility. Pseudocode for the training file is as follows:  
```
for e in epochs:    
    instantiate environment as env
    while trial not terminated:
        initialize env
        check if agent at intersection
        if at intersection:
            move forward
        else:
            if random int < epsilon:
                explore by choosing an action at random
            else:
                chose the policy model
        experience replay to get previous experiences
        update the model based on experience replay
```
Pseudocode for the training file is:  
```
load pre-trained model
for t in trials:
    instantiate environment
    while trial not terminated:
        initialize environment
        check if agent at intersection
        if at intersection:
            use q-values from model to determine action
        else:
            move agent forward
        keep count of values used for evaluation metrics
```

## Running the code  
Training:  
`$ python3 qlearn_crossing.py`    
use the --help or -h flag to see available options.

Testing the trained model:  
`$ python3 test_drive.py`  
use the --help or -h flag to see available options.
