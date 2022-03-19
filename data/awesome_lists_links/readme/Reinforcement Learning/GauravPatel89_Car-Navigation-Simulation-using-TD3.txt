# Project EndGame: Car Navigation using Deep Reinforcement Learning

## Problem Statement:  
Teach car to reach different goals on the city map while traveling on road using **Twin Delayed DDPG (TD3)** algorithm. 

### Submission:

[![EndGame Final Submission](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/endGameVideoSnapShot.png)](https://youtu.be/77mfBYC8f0g)

### How to generate above video?

To generate output shown in above video, car needs to first learn to navigate the map. Car can then make use of this knowledge to navigate roads and hit the targets as shown in the video.

#### How to train your car?
Just like humans Car needs to go through tons of experiences, learn which actions are rewarding and which are penalizing, experiment, make mistakes, improvise. It's a lot. But its doable using Deep Reinforcement Learning.

In this submission we have used one of the most powerful RL algorithm, [**Twin Delayed DDPG**](https://arxiv.org/pdf/1802.09477.pdf) aka **TD3**. Detailed explanation on working of this algorithm can be found in previous [session submissions](https://github.com/GauravPatel89/EVA-Track3-Assignments/tree/master/P2S9) and at references listed at the end of this page.   

Coming back to implementation, Car's navigation knowledge lies in TD3 algorithm's Actor and Critic models. To train these models there are two options

1. Train on local machine  
- Download entire EndGame [folder](https://github.com/GauravPatel89/EVA-Track3-Assignments/tree/master/EndGame)
- cd to 'EndGame' directory   

        cd path-to-EndGame-folder
        
- Run [endGameTD3.py](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameTD3.py) 

        python endGameTD3.py

2. Train on Google [Colab](https://colab.research.google.com/)
- Upload [endGameTD3.ipynb](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameTD3.ipynb) to Colab

- Create a folder at '/content/drive/My Drive/endGameTD3'

- upload files ('car.png','citymap.png','MASK1.png','endGameEnv.py','endGameModels.py','endGameUtilities.py')

- Run all (option under 'Runtime' in toolbar)

The training process will go on for a while. Meanwhile intermediate training information and evaluations will be shown to convey how training is progressing. Training will periodically store trained Actor, Critic models to 'pytorch_models' directory. 

Google Colab file for our training can be accessed [here.](https://colab.research.google.com/drive/1S3kT0hJlK4Uzh10DrAFE55l9OtZAbLyc?usp=sharing)

Google drive link for our training run directory can be found [here.](https://drive.google.com/drive/folders/16r-DNY4tIaqXO4Be8AINRIiIk3DcfohM?usp=sharing).

Best trained models for our run can be found [here.](https://github.com/GauravPatel89/EVA-Track3-Assignments/tree/master/EndGame/pytorch_models)


#### How to Run your Car?

Now that our Car has learned so much, we must see how it performs. This step must be run on local machine as follows.
- cd to 'EndGame' directory   

        cd path-to-EndGame-folder


- Copy Actor, Critic models to be evaluated to 'pytorch_models' with names 'TD3_carEndGameEnv_0_actor.pth' and 'TD3_carEndGameEnv_0_critic.pth' respectively.
- Run [endGameInference.py](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameInference.py)

        python endGameInference.py

- 'endGameInference.py' provides options for live car running view, video generation and average reward visualization.

### Under The Hood

#### The Environment

To simulate the car navigation task an environment has been defined. Environment definition is similar to standard Gym environments. Important functions like step(),reset(),render() have been provided for ease of use. This way the training algorithm need not worry about car movement,visualization, reward generations etc. Actor has to just query the environment for current state and based on that provide action to the environment. Environment in turn takes provided action and generates next state and also informs Actor about reward for that step. 

Some of the important components of the Environment are explained below.

##### State space:
State for the environment should be such that it satisfies markov model, meaning at any point of time the state should be able to represent environment's current setup irrespective of past actions and states. In simple terms we must define 'State' such that it conveys to the model all the information about the Environment in order to take an appropriate action to achieve the specified target.

For our problem, the target can be simplified into 2 tasks.  
1. Stay on the Road  
2. Reach the Goal

For achieving first task we must define our state such that model can ascertain whether it is on road and if not how and where to turn to get back on road.  
For second task, we must have have information as to how far is the Goal and where does it lie wrt the car.  
Keeping this in mind we have 4 components in our 'State'

1. Current state image:  
Current state image for the environment is a cropped view of road network in car's front view i.e. how car is viewing the area around it. In this view car is always facing front but area around it changes as car navigates as shown below.    

    <p align="center">
      <img width="100" height="100" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/stateImage.gif">
    </p>  
    
       
    Currently crop size has been selected to be 40x40 . This image is estimated by first cropping an area twice the required crop size, rotating it by (90-car.angle) then again cropping it to required crop size. One such example is illustrated below. Car angle is assumed to be 10 degrees.
    
    <p align="center">
      <img width="800" height="800" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/StateFig1.bmp">
    </p>
    
    <p align="center">
      <img width="800" height="220" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/StateFig2.bmp">
    </p> 


2. Normalized Distance to Goal:  
This value corresponds to euclidean distance of Goal from car, normalized by max possible goal distance (diagonal distance of citymap image).

3. Goal Orientation:  
This value corresponds orientation of Goal wrt to Car's current inclination angle. It is calculated as angular difference between vector1, joining car location and goal location, and vector2, representing Car's pointing direction.

4. -ve Goal Orientation: 
This value is same as previous value but with -ve sign.

##### Action space:  
Action space for the environment defines kind of actions environment allows an Actor to take. For our environment action space is 1 dimensional i.e. just one value, 'angle of rotation'. 

For each of the env.step(action) execution, car is first rotated by given 'action' and then displaced as per its velocity along 'car.angle'.

Max size of this rotation has been limited to 5 degrees thus during any step car can rotate by maximum of -5.0 or 5.0 degrees.

##### An Episode:
Runs of environment are classified as episodes. In each episode, car tries to reach 3 number of randomly selected goals. Episode is ended based on 3 criteria.

1. All the 3 Goals achieved.
2. Car is at the boundary and its position has not changed for last 50 steps
3. Maximum allowed steps (5000) have been taken without encountering previous two conditions.

##### Reward System:
The reward system plays most important role in conveying to the Actor which actions in particular state are rewarding and which are not. 

For our environment we have following reward system.
1. On reaching a Goal location: +50.0
2. On touching walls : -10.0
3. For being on road and moving towards Goal: 0.5
4. For being on road but moving away from Goal (Living Penalty): -0.9.
5. For not being on road: -5.0

##### Goal Selection:
The environment provides 3 Goals during each episode. In order to prevent Actor from memorizing (Overfitting) path to Goals it is important to randomize them. In our environment 3 Goals are selected at random during each episode from a list of 10 Goals. 

#### Twin Delayed DDPG (TD3)
At the heart Navigation learning is Twin Delayed DDPG (TD3) algorithm. It's implementation has been explained in detail in previous session [here](https://github.com/GauravPatel89/EVA-Track3-Assignments/tree/master/P2S9). 

TD3 consists of 2 types of Neural Networks. 
1. **Actor**  
Task of an Actor is to predict an 'Action' for given 'State'. In our implementation Actor network is as shown below.

<p align="center">
      <img width="360" height="840" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/Actor.png">
    </p>

2. **Critic**
Task of a Critic is to predict a Q-Value for given 'State'  and 'Action' (Generated by Actor). In our implementation Critic network is as shown below. 

<p align="center">
      <img width="360" height="840" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/Critic.png">
    </p>

### CodeBase

#### Files:
1. 'endGameEnv.py' :  
This file contains Environment Class definition for simulation of the Car Navigation Game. For ease of use the environment class has been defined on the lines of standard Gym environments i.e. class has methods like reset(),step(),render(),sampleActionSpace(). 

2. 'endGameModels.py' :  
This file contains definition of Actor and Critic DNNs, ReplayBuffer used for storing step transitions and TD3 class which implements TD3 algorithm.

3. 'endGameInference.py' :  
This file contains code for evaluating the trained models. This is done by instantiating TD3 class object, loading saved Actor and Critic models and repeatedly generating action and taking action on defined environment to generate visualization.

4. 'endGameUtilities.py' :  
This file contains some of the utilities used by other files.

5. 'endGameEnvAllGoals.py' :  
This file has Environment definition similar to 'endGameEnv.py'. Difference is 'endGameEnv.py' environment's episode runs for only 3 random goal values while environment defined in this file runs the episode untill all the goals in the goalList have been achieved or car gets stuck to boundaries. This file is useful for evaluating how model is working for all the goal values. It was used for generating the submission video.

6. 'endgameTD3.py' :  
This file contains code for TD3 training. It combines components in other files to create the main training loop.

7. 'endGameTD3.ipynb'
This is the Google Colab file for TD3 training. 'endgameTD3.py' is simple .py version of this file. This file can be accessed on Google Colab [here.](https://colab.research.google.com/drive/1S3kT0hJlK4Uzh10DrAFE55l9OtZAbLyc?usp=sharing)

#### Directories
1. 'pytorch_models':  
This directory contains best saved models for Actor and Critic. Example code for evaluating and generating a video for this models can be found in ['endGameInference.py'](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameInference.py). 

2. 'images':  
This directory contains image files used by the carEndgameEnv environment. It has following files
    
    a. 'car.png': Used for visualization of Car on city map. 
    b. 'citymap.png' : Used as city map
    c. 'MASK1.png': Used for representing road and non-road pixels in city map

3. 'Figures':  
This directory contains support files for README.md .

4. 'pytorch_models':
This directory contains best trained Actor, Critic models used for generating submission video.

    
### References  
1. [Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.](https://arxiv.org/pdf/1802.09477.pdf)
2. [OpenAI — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html) 
3. [Solving-CarRacing-with-DDPG](https://github.com/lzhan144/Solving-CarRacing-with-DDPG/blob/master/TD3.py)
4. [TD3: Learning To Run With AI](https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93)
    

