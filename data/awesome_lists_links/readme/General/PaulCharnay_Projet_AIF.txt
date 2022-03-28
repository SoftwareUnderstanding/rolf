# Reinforcement Learning: solving MsPacman with Actor-Critic algorithms
This repository contains Python files to solve MsPacman using Actor-Critic algorithms (A2C and ACER).

It was created by the following students from INSA Toulouse:
- Camille Bichet
- Paul Charnay
- Oumaima Dahan
- Louis Delvaux
- Emmeline Monédières

## Useful links
- Actor-Critic papers: 
    - A2C: https://arxiv.org/abs/1602.01783
    - ACER: https://arxiv.org/abs/1611.01224

- Baselines blog on Actor-Critic: https://blog.openai.com/baselines-acktr-a2c/

- MsPacman on OpenAI Gym: https://gym.openai.com/envs/MsPacman-v0/

- Stable-baselines repository: https://github.com/hill-a/stable-baselines

## Files
- `train_a2c.py`: Train an A2C model for MsPacman.
- `train_acer.py`: Train an A2C model for MsPacman.
- `video_a2c.py`: Record a video of the trained A2C agent playing MsPacman.
- `video_acer.py`: Record a video of the trained ACER agent playing MsPacman.
- `show_videos.ipynb`: notebook showing the recorded videos.

## Requirements
The following python libraries are required:
- gym
- stable-baselines