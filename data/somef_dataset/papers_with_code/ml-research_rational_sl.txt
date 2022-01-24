[![ArXiv Badge](https://img.shields.io/badge/Paper-arXiv-blue.svg)](https://arxiv.org/abs/2102.09407)

# Rational Supervised Learning
Rational Networks in Supervized Learning such as MNIST, CIFAR and Imagenet Classification Tasks.

Rational functions outperformes every non-learnable one (*cf. [Padé Activation Units: ...](https://arxiv.org/pdf/1907.06732.pdf)*).

  ![sl_score](./images/sl_score.png)

Rational are also here used for lesioning. They replace Residual Blocks in a ResNet101:

|  Eval      | Lesion   |  L2.B3  |  L3.B13 |  L3.B19 |  L4.B2 |
|  ----      | ------   |  -----  |  ------ |  ------ |  ----- |
| training   | [Standard](https://arxiv.org/pdf/1605.06431.pdf) |  100.9  |  120.2  |  90.5   |  58.9  |  
|            | [Rational](https://arxiv.org/pdf/2102.09407.pdf) |**101.1**|**120.3**|**104.0**|**91.1**|
| testing    | Standard | **93.1**| 102.0   |   97.1  |  81.7  |  
|            | Rational |   90.5  |**102.6**| **97.6**|**85.3**|
|% dropped params |     | 0.63    | 2.51    | 2.51    | 10.0   |

# Dependencies
- Python 3.6+
- [PyTorch]()
- [rational-activations](https://github.com/ml-research/rational_activations)

# Using Rational Neural Networks
If you want to use (Recurrent) Rational Networks, an additional README is provided in each folder to explain how to train the networks, make score tables, plots, ...etc

# Related Repo
[Rational RL](https://github.com/ml-research/rational_rl) - for (Recurrent) Rational Networks on Reinforcement Learning
