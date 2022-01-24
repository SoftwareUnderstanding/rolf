# Super Mario Bros

An implementation of the Double Deep Q Network (DDQN) to learn to play Super Mario Bros.

It takes ~30 hours of training on a GPU to get a 60% win rate. Training for more than 30 hours doesn't seem to increase that win rate.

Letting the agent explore for a longer period of time by slowing down epsilon decay might improve the win rate?

![demo](/demo.gif)

## Set Up 
```
# Create virtual environment 
python3 -m venv env

# Activate virtual env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# To deactivate virtual env
deactivate
```

## Usage
```
# Train the agent
./mario.py train

# Watch mario play 
./mario.py play
```

## Training graphs
You can see that the reward peak at episode 50,000 (30 hours of training) then stagnates.

![distance](/data/ep_rewards_plot.png)

More plots can be found at `data/`

## Resources
* Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602
* Deep Reinforcement Learning with Double Q-learning: https://arxiv.org/abs/1509.06461
* Intro to Reinforcement Learning with David Silver: https://youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB
* Neural Networks from 3Blue1Brown: https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 
* Conv Nets: A Modular Perspective: https://colah.github.io/posts/2014-07-Conv-Nets-Modular/
