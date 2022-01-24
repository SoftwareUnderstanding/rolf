# A3Thor - object driven navigation in indoor scene using A3C

This is final project of final semester in robotic - INT 3409 1 

We using A3C - Asynchronous advantage actor critic algorithm to train an agent navigating in side simulated environment ai2thor

## Overview

This project includes implementations of A3C in `./A3C/a3c.py`

to train the model, using:

`python main.py --is_ai2thor`

to visualize result, using:

`python --is_ai2thor --critic_path */A3C/model/critic-model* --actor_path */A3C/model/actor-model*`

The default training parameter is 5000 episodes, 5 threads
## Installation

Clone this repository:

Install Python dependencies:

`pip install -r requirements.txt`

Highly recommend to install tensorflow using conda:

`conda install tensorflow-gpu`
## Project description

* this project using gym-style interface of ai2thor environment
* objective is simply picking an apple in kitchen environment - FloorPlan28
* observation space is first-view RGB 128x128 image from agent's camera
* maximum step in this project is 500
* reward fuction: 
    * -0.01 each time step
    * 1 if agent can pick an apple, the env than terminate
    * 0.01 if agent saw an apple (has been removed in latest code) 
* a pre-train mobilenet-v2 model on image-net is used an feature extractor for later dense layer both actor and critic model
* actor optimizer using Advantages + Entropy term to encourage exploration (https://arxiv.org/abs/1602.01783)

# Training 

* this project trained on xenon E5-2667v2 + GTX1070, with 1,444,234 parameters for actor and 1,443,073 params for critic model
the objective is simple so that the model converge very fast, detail log and trained model in ` ./A3C/`

This project greatly thanks to material:
- [Gym-style ai2thor environment](https://github.com/TheMTank/cups-rl)
- [Collection of deep rl algorithms in keras](https://github.com/germain-hug/Deep-RL-Keras)
