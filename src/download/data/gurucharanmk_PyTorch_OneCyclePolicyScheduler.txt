# PyTorch_OneCyclePolicyScheduler (Work In Progress)

### Overview

PyTorch implementation of One cycle policy as described in Leslie Smith's paper: https://arxiv.org/abs/1803.09820.

Approach is the combination of gradually increasing learning rate (to optimal learning rate as per https://arxiv.org/abs/1506.01186 by same author), gradually decreasing the momentum during the first half of the cycle, then gradually decreasing the learning rate and increasing the momentum during the latter half of the cycle to achieve "super convergence".

### ToDo
- Verify initial implementation as mentioned in heder of this doc it is wip
- Integrate [leaning rate finder](https://github.com/gurucharanmk/PyTorch_LearningRateFinder)
- Integrate weight decay parameter, as paper sugests it needs to be constant through out of this training.
- Source code clean-up, comments and docstrings !!
