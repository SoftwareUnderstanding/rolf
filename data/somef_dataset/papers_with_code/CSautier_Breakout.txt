# Pytorch multiprocessing PPO implementation playing Breakout

## How it works

The optimization is a standard PPO implementation, however the point was to push the limits of what a limited computer 
could do in reinforcement learning. Thus I use multiple processes to play the game and gather experiences. However, if 
multiples processes try to access a single gpu, most of the computation time will be lost to each process waiting for 
their turn on the gpu, rather than actually playing the game, resulting in a very limited speedup between multiprocessed 
and not multiprocessed algorithms. Furthermore it necessitated the net to be copied on multiple processes, wich was very 
VRAM consuming. \
\
This algorithm works differently:
* multiple processes play the game
* a single process has access to the gpu
* when a playing process requires the gpu, it sends the operation to execute to the gpu process, and the gpu process 
sends back the result

This way, the training can be around twice as fast for a computer with a single GPU compared to a naive multiprocessed 
PPO

## Requirements

* Pytorch
* Numpy
* gym (Atari)
* a few standard libraries such as argparse, time, os
* There is no guarantee this will work in python 2, or without a GPU
* around 2Gb of RAM for each core of your CPU with the recommended number of workers

## How to begin the training

* Clone this repository: `git clone https://github.com/CSautier/Breakout`
* Launch the game in a shell: `python Breakout.py`
* If you'd prefer a faster training, you can deactivate the visualization: `python Breakout.py --render False`

## Useful resources

* https://openai.com/
* https://arxiv.org/pdf/1707.06347.pdf

**Feel free to use as much of this code as you want but mention my github if you found this useful**.  
**For more information, you can contact me on my github.**
