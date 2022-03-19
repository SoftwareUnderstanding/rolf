# game_playing
Simple game playing with reinforcement learning and vowpal wabbit - still work in progress


## Instructions to Install:
**1. Install python / vw / openAI gym development tools and dependancies:**
 * On linux / centos:
 * sudo yum groupinstall "Development Tools"
 * sudo yum -y install gcc-c++ python-devel atlas-sse3-devel lapack-devel gcc-gfortran
 * sudo yum install boost-devel zlib-devel cmake
 * On Mac:
 * brew update
 * brew install boost --with-python
 * brew install boost-python
 * brew install cmake

**2. Create a folder:**
* sudo mkdir /usr/local/myproject
* cd /usr/local/myproject
* sudo chmod -R 777 myproject/

**3. Install pip and virtualenv (skip if you already have it installed):**
* sudo curl -o /tmp/ez_setup.py https://bootstrap.pypa.io/ez_setup.py
* sudo /usr/bin/python /tmp/ez_setup.py 
* sudo /usr/bin/easy_install pip 
* sudo pip install virtualenv
* cd myproject
* virtualenv venv_ml
* source /usr/local/myproject/venv_ml/bin/activate

**4. Install python packages in virtual environment:**
* pip install numpy
* pip install mmh3
* pip install gym
* pip install gym['atari']	
* pip install pyaml		
* pip install pandas	
* pip install scipy

**5. Install vw with python:**
* Either just 'pip install vowpalwabbit'

Or,

* git clone https://github.com/JohnLangford/vowpal_wabbit
* cd vowpal_wabbit
* make
* sudo make install
* make python
* cd /usr/local/myproject/vowpalwabbit/python
* sudo cp pylibvw.so /usr/local/myproject/venv_ml/lib/python2.7/lib-dynload/
* cp -R vowpalwabbit/ /usr/local/myproject/venv_ml/lib/python2.7/site-packages/vowpalwabbit

**6. Instructions to run training**
* git clone https://github.com/joshiatul/game_playing

cd game_playing and then run (set config yaml parameters):
* python simulate_environment.py --config example_config.yaml --train
* python simulate_environment.py --config example_config.yaml --test


## Status:
* Solves gridworld (4x4, 5x5, 6x6 so far, does better than random on 7x7) using asynchronous methods as descibed in (2) (also supports experience-replay described in (1) but works 10x slower wrt (2))

## TODOs:
1. Train / try to solve OPENAI gym games - pong / breakout
2. Experiment with actor-critic methods
3. Try to implement policy gradients and experiment with them
4. Fix blackjack implementation


## References:
1. Mnih et el 2013, Playing Atari with Deep Reinforcement Learning, NIPS Deep Learning Workshop 2013 (http://arxiv.org/abs/1312.5602)
2. Mnih et el 2016, Asynchronous Methods for Deep Reinforcement Learning, Proceedings of the 33rd ICML, New York, NY, USA, 2016 (http://arxiv.org/abs/1602.01783)
3. David Silver's "Reinforcement Learning" lecture videos (https://www.youtube.com/watch?v=2pWv7GOvuf0)
4. https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
