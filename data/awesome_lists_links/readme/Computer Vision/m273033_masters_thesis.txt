# Transfer Learning in Automated Car Driving
**Goal** : Train an autonomous car driving RL agent in one environment and transfer that knowledge to another environment and achieve state-of-the-art driving performance in it.

## Semantic Segmentation
Adapted tensorflow implementation of ENET (https://arxiv.org/pdf/1606.02147.pdf) (https://github.com/kwotsin/TensorFlow-ENet)
- Modified to support the transfer learning
- Transfer learning between virtual and real world environments
- Virtual environment : CARLA simulator
- Real world environment : Cityscapes Dataset

## RL Agent
rl_coach to communicate between CARLA and python (https://github.com/NervanaSystems/coach)
- Modified presets algorithms and CARLA environment

### Requirements
- CARLA simulator (https://github.com/carla-simulator/carla)
- Cityscapes Dataset (https://www.cityscapes-dataset.com)
- rl_coach (https://github.com/NervanaSystems/coach)
