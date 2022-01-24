# TP4-DDPG
Project for INF8225 TP4

DDPG implementation with pytorch

Paper: https://arxiv.org/abs/1509.02971


# Quickstart
Clone repository and create venv with:
```
python3.7 -m virtualenv <env_name>
source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```
Run pre-trained models with:

```
python run_trained.py <env>
```
using "p","mc" or "ll" for the different environments
Pendulum Swing up, Mountain Car and Lunar Lander.

New models can be trained running

```
python sandbox.py
```
Rendering can be enabled/disabled by pressing "Enter" in the console.

# Performance on OpenAI Environments

## Pendulum-v0 after 125 Episodes

![Pendulum-v0](gifs/pend.gif)

## MountainCarContinuous-v0 after 150 Episodes

![MountainCarContinuous-v0](gifs/mc.gif)

## LunarLanderContinuous-v2 after 200 Episodes

![LunarLanderContinuous-v2](gifs/ll.gif)
