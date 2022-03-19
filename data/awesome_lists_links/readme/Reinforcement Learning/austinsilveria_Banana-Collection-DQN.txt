# Unity Banana Collection

### Environment Details

The goal of this environment is to train an agent to navigate and collect bananas in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Installation

The environment provided in the repository is for Windows (64-bit).

If you need a different version, download the environment from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
#### Required Dependencies:
    Python 3.6
    Unity Agents
    PyTorch
    Numpy
    Matplotlib
    Jupyter (Optional, python files are provided if you wish to run from command line)
    
### Instructions

First, be sure to change the file path of the Unity Environment in the 'Instantiating the environment' cell to your environment's location.

If you wish to train your own agent, the implementation in 'Banana Collection Double DQN PER.ipynb' can be used by running the Jupyter cells directly in order. This includes a Double DQN and Prioritized Experience Replay. If you would like to see a pre-trained agent, skip the cell 'Initialize agent that takes actions and learns from the environment', and run 'Watch trained agent' to load parameters for the agent watch it navigate the world.

To run the implementation without Prioritized Epxerience Replay or investigate the difference between uniform sampling and Priortized Experience Replay, 'Banana Collection Double DQN.ipynb' is provided as well.

### Troubleshooting

If the environment does not respond, force quit the environment window and under 'Kernel' in the Jupyter Notebook, click 'Restart & Clear Output'. Running the cells again will reinstantiate a fresh environment.

### Sources

DQN Model and Agent adapted from: https://github.com/udacity/deep-reinforcement-learning

DQN: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Double DQN: https://arxiv.org/pdf/1509.06461.pdf

Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952.pdf

Sum Tree: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/

Dueling DQN: https://arxiv.org/pdf/1511.06581.pdf