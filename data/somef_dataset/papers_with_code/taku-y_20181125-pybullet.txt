# 20181125-pybullet

## Install python 3.6 with homebrew
Python 3.7 have a problem when installing tensorflow 
(https://github.com/tensorflow/tensorflow/issues/20444).

```bash
# See https://apple.stackexchange.com/questions/329187
$ brew install \
  https://raw.githubusercontent.com/Homebrew/homebrew-core/\
  f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb
```

## venv
- https://qiita.com/fiftystorm36/items/b2fd47cf32c7694adc2e

```
$ cd $WORKDIR
$ python3 -m venv pybullet-env
$ source pybullet-env/bin/activate
$ pip install tensorflow
$ pip install gym
$ git clone https://github.com/openai/baselines.git
$ cd baselines
$ pip install -e .
$ cd ..
$ pip install pybullet
$ pip install ruamel-yaml
```

## Check pybullet
```bash
$ cd pybullet-env/lib/python3.6/site-packages/pybullet_envs/examples
$ python kukaGymEnvTest.py
$ python kukaCamGymEnvTest.py # much slower
```

## References
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015, June). [Trust Region Policy Optimization.](http://proceedings.mlr.press/v37/schulman15.pdf) In ICML, 2015
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal policy optimization algorithms.](https://arxiv.org/abs/1707.06347) arXiv preprint arXiv:1707.06347.
- [Approximately optimal approximate reinforcement learning](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjPytbijb_eAhUZfnAKHfAtDzEQFjAAegQICRAC&url=https%3A%2F%2Fpeople.eecs.berkeley.edu%2F~pabbeel%2Fcs287-fa09%2Freadings%2FKakadeLangford-icml2002.pdf&usg=AOvVaw1lMj6AB90nJbKT-5pNcQ9P)
- [Reinforcement Learning: An Introduction](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiT_9bN-77eAhXYdXAKHetgAiQQFjABegQIBRAC&url=http%3A%2F%2Fincompleteideas.net%2Fbook%2Fbookdraft2017nov5.pdf&usg=AOvVaw00kFmqVbFSdkU3PTkJMJrO)
  - Chapter 13 
Policy Gradient Methods
  - 13.2 The Policy Gradient Theorem
  - 13.3 REINFORCE: Monte Carlo Policy Gradient
  - 13.4 REINFORCE with Baseline
  - 13.5 Actor–Critic Methods
- [Understanding RL: The Bellman Equations](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/)
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, June). [Deterministic policy gradient algorithms.](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf) In ICML, 2014.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). [Continuous control with deep reinforcement learning.](https://arxiv.org/abs/1509.02971) arXiv preprint arXiv:1509.02971.
- [OpenAI Gym 入門](https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16)
- [[Python] Keras-RLで簡単に強化学習(DQN)を試す](https://qiita.com/inoory/items/e63ade6f21766c7c2393)
- [OpenAI GymでFXのトレーディング環境を構築する](https://qiita.com/hide-tono/items/bb9691477831e48f0989)
- Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., & Vanhoucke, V. (2018).
  [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots.](https://arxiv.org/abs/1804.10332) arXiv 
  preprint arXiv:1804.10332.

## Progress
- baselinesによる動作はバグのため失敗。
  `TypeError: learn() missing 1 required positional argument: 'network'`
  というエラー。
- [Tensorflow agents PPO](https://github.com/google-research/batch-ppo)による
動作確認はできた。ただし訓練のみ。警告が大量に表示されるので消したい。 `pendulum` という
名前のディレクトリが作成される。Configurationは `pybullet_envs/agents/configs.py`
の中で設定されている。
