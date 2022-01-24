# OpenAi ROS integration

![logo](https://media.giphy.com/media/YkBoQvvVPcSGF4a9Mf/giphy.gif)

## Setup
1. Clone the repo in your ros workspace.
``` git clone https://github.com/dsapandora/s_cera```
2. The issue with **OPENAI RETRO GYM** and **ROS** is tha OPENAI work only with *python 3.5*  and ROS can hardly work *python 2.7* [StackOverflow](https://stackoverflow.com/questions/49758578/installation-guide-for-ros-kinetic-with-python-3-5-on-ubuntu-16-04) 
3. Inside the s_cera, create a virtual env enviroment used for python 3.5.
```
virtualenv -p python3  env
source env/bin/activate
pip3 install gym-retro
pip3 install opencv-python
pip3 install pygame
pip3 install imutils
pip3 install scipy
pip3 install pyyaml
pip3 install catkin_pkg
pip3 install rospkg
mkdir roms
chmod +x retro_gym_server.py
chmod +x viewer.py
```
4. This plugin is test with sonic the headhog from sega genesis,but we don't distribute the rom in this repo. But the rom is easy to find in the internet. But If you still have issue to find it, just let me know. The rom must be places in the rom folder. To see wich rom are compatible and more info about retro gym just follow the link [gym retro repository](https://github.com/openai/retro)

5. Import the rom with the following command: ```python3 -m retro.import roms``` It must be executed in the s_cera folder.
7. In your workspace execute: ```catkin_make``` 

## Usage
![Imgur](https://i.imgur.com/GtFcaIG.png)

1. The server will run in python 3 using the python3 env, the reason behind is the following [issues](https://stackoverflow.com/questions/43019951/after-install-ros-kinetic-cannot-import-opencv), so to run the server a export need that change the issue with opencv must run before the server execution.

```
export PYTHONPATH="<S_CERA_FOLDER_PATH>/env/lib/python3.5/site-packages:$PYTHONPATH"
rosrun s_cera retro_gym_server.py
```

2. To execute the client you must open another console, and run:
``` rosrun s_cera viewer.py```

This will open a pygame view that will allow you to control the agent in the enviroment. Using the keyboard. 


**Because the game is tested with sonic, i didn't intend to map every sega genesis button. But you can do it if you want.***


## Topics
![Imgur](https://i.imgur.com/csP34le.png)

The server publish a compressed image topic **aka a numpy matrix** named: **world_observation/image_raw** and is subscribed to a topic named: **world_observation/cmd_vel** that is a *Twist* message. Sonic is moved with linear.x and linear.y.
The angular and the linear.z is open for any change that you want to apply.  

* **Linear.x** move the sonic agent, 1 forward, -1 backward.
* **Linear.y** move the sonic agent 1 jump, -1 crunch. 

The viewer publish  **world_observation/cmd_vel**  topic mapped from the **key events in pygame**. and is subscribed to  **world_observation/image_raw** so it will desplay instantaneously what is happening in the server. 

You can use this topic for train sonic, or to test another kind of roms. 

## Future use of this repo (ignore for now.)
open ai
https://blog.openai.com/retro-contest/
https://contest.openai.com/2018-1/
https://arxiv.org/pdf/1804.03720.pdf
https://github.com/openai/retro


retro ppo baselines
https://blog.openai.com/openai-baselines-ppo/
https://github.com/openai/retro-baselines


Rainbow: Combining Improvements in Deep Reinforcement Learning
https://arxiv.org/abs/1710.02298

retro contest retrospective
https://blog.openai.com/first-retro-contest-retrospective/


META LEARNING SHARED HIERARCHIES
https://arxiv.org/pdf/1710.09767.pdf)


OpenAI retro reports
https://medium.com/@olegmrk/openai-retro-contest-report-b870bfd014e0

Policy distillation
https://arxiv.org/pdf/1511.06295.pdf


jerk agent
https://github.com/olegmyrk/retro-rl/blob/master/jerk_agent.py


dylan world model
https://dylandjian.github.io/world-models/
https://github.com/dylandjian/retro-contest-sonic


Jaan altosar
https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

conscale
http://www.conscious-robots.com/consscale/


human like behaviour
https://www.sciencedirect.com/science/article/pii/S0957417414002759?via%3Dihub


Soar cognitive
https://github.com/SoarGroup/Soar

cera
http://www.conscious-robots.com/es/tag/cranium/
