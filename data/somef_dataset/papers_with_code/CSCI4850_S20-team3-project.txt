# S20-team3-project

Requirements:

    python 3.7
    virtualenv
        
And all their dependencies. These are available in the requirements.txt

## Automatic setup

A bash file, setup.sh, has been provided to setup the virtual environment and edit the Galaga gym-retro environment for you. To use this, ensure you have virtualenv and python3.7 installed, then simply run the script:

    ./setup.sh

Then, you must activate the virtual environment:

    source venv/bin/activate

Once the virtual environment is active, if you would like to see a short demo of each network, run the demo.py file:

    ./demo/demo.py

Or to experience training, step into either SingleQ/ or DoubleQ/, and further into either Uniform/ or Prioritized/ and use:

    ./Galaga.py
    
To watch the gameplay, simply use:

    ./Galaga.py --play

## Manual Setup

Create a 3.7 virtualenv

    virtualenv --python=/path/to/python3.7 venv # /usr/bin/python3.7 for example

start the virtual environment- ensure that you are in your local repository

    source venv/bin/activate
    
install requirements:
    
    pip3 install -r requirements.txt
    
edit the documents noted within setup.sh to meet the environment requirements.
    
Finally, import the rom:

    python3 -m retro.import 'Galaga - Demons of Death (USA).nes'

## Action Space
As designated by Arcade Learning Environment Technical Manual[^1], we have
selected six possible actions to control the game:

- Do-Nothing (0)
- Left       (3)
- Right      (6)
- Fire       (9)
- Left-Fire  (12)
- Right-Fire (15)

This exists as our "small action space", where we restrict the network's
available actions to prevent it from "gaming" the reward system or getting
stuck. However, the default action space for Galaga is also the same as this
small action space, simply repeating as per the atari-py implementation manual[^1].

## Implementations
Our full plan is to implement four different versions of the Deep Q-Learning
architecture. Utilizing research on the Memory Replay component, we intend to
compare the benefit of the Prioritized Memory Replay[^2] to that of the Uniform
Memory Replay that we have interpreted as the norm of Deep Q-Learning.

Further, after completing these original two implementations, we plan to
implement both Memory Replay methodologies within a Double Q-Learning
implementation[^3]. This will allow us to compare the viability of the
Prioritized Memory Replay within both environments, and compare the viability of
the Double Q-Learning implementation.

Parameters and training regiments will be standardized across all
implementations to ensure reproducability and comparability.

## References
1: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf

2: https://arxiv.org/abs/1511.05952

3: https://arxiv.org/abs/1509.06461f
