# StarCrackAI
StarCraft is a real-time strategy(RTS) game that combines fast-paced micro-actions with the need for high-level planning and execution. Over the previous two decades, StarCraft I and II have been pioneering and enduring e-sports, with millions of casual and highly competitive professional players. In Aug 2017, Deepmind published a paper "StarCraftII: A New Challenge for Reinforcement Learning" giving some intuitions about how to train AI to defeat top human players. Right now AI player is good at single-agent, single-player interacting, but poor performance on multi-agent, with multiple-players interacting. We are trying to build a system supervised learning from replays and reinforcement learning from baseline agents, mimic human player actions and ... You can solo or 2v1 against AI, would be a lot of fun!

## Resources
* Blizzard's SC2 Machine Learning API: https://github.com/Blizzard/s2client-proto
* Openmind's python wrapper component: https://github.com/deepmind/pysc2

## Algorithms:
* OpenAI's baselines: https://blog.openai.com/openai-baselines-dqn/   &&   https://github.com/openai/baselines
(Note: first thing i tried is baselines DQN, but while i was trying, their github is fail to build for over a month, they just fix it...)
* PPO: https://arxiv.org/abs/1707.06347

* Others: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
(Note: I found a very good tutorial while i was digging into PPO algorithm, basically have all codes already includes)

## Get Started!
### Installation
IMPORTANT NOTE:  PySC2 2.0.1 must use game client v4.1.2 or above

### Smoke Test
* Installation Verify: python -m pysc2.bin.agent --map Simple64
* Replay Verify: python -m pysc2.bin.play --replay <path-to-replay>
* List Maps: python -m pysc2.bin.map_list

### RL smart agents


### All existing related resourses(But all resources is out of date)
* https://github.com/greentfrapp/pysc2-RLagents
* https://github.com/Inoryy/pysc2-rl-agent
* https://github.com/xhujoy/pysc2-agents
* https://github.com/chris-chris/pysc2-examples
* https://github.com/simonmeister/pysc2-rl-agents
* https://github.com/chagmgang/pysc2_rl

* https://zhuanlan.zhihu.com/p/29246185?group_id=890682069733232640
