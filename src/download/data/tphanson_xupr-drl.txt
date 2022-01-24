# xupr-drl
Manually construct a comprehensive DRL model which use all state-of-the-art techniques

## Environment

We use [OhmniInSpacev0](https://github.com/tphanson/tf-agent-labs/tree/c51)

## Checklist
- [x] Q-Learning
- [x] Deep Q-Learning (Add CNN)
- [x] Deep Recurrent Q-Learning (Add RNN)
- [x] Prioritized Experience Replay
- [x] Double Q-Learning
- [x] Dueling Networks
- [x] Multi-steps Learning
- [x] Distributional Reinforcement Learning
- [ ] Distributed Learning

## References

[1] Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

[2] Horgan, Dan, et al. "Distributed prioritized experience replay." arXiv preprint arXiv:1803.00933 (2018).

[3] Kapturowski, Steven, et al. "Recurrent experience replay in distributed reinforcement learning." International conference on learning representations. 2018.

[4] Tomassoli, Massimiliano. “Distributional RL.” Simple Machine Learning, mtomassoli.github.io/2017/12/08/distributional_rl/.

## Unsorted Refs

[1] https://www.researchgate.net/publication/335659932_Navigation_in_Unknown_Dynamic_Environments_Based_on_Deep_Reinforcement_Learning

[2] http://ras.papercept.net/images/temp/IROS/files/0386.pdf

[3] https://arxiv.org/pdf/2005.13857.pdf

[4] https://arxiv.org/abs/1511.05952

## Convenient commands

```
# Download model

rm -rf ~/Desktop/xupr-drl/models/ && scp -P 14400 -r tuphan@192.168.123.58:/home/tuphan/xupr-drl/models ~/Desktop/xupr-drl/
```