# maml
MAML - Coursework of Machine Learning MPhil at Cambridge
- This repository aims to reproduce the paper "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" - https://arxiv.org/abs/1703.03400
- This is MLMI4 Advanced Machine Learning Coursework
## Results
### Sinunoid Regression
- look at regression_eval.ipynb
### Image Classification
- Omniglot

|   MAML   |  1-shot  |  5-shot  |
|:--------:|:--------:|:--------:|
|   5-way  | 94.7±0.5 | 98.8±0.1 |
|  20-way  | 83.4±0.4 | 94.1±0.2 |

- MiniImagenet

| 5-way |   1-shot   |   5-shot   |
|:-----:|:----------:|:----------:|
|  MAML | 45.14±1.99 | 62.06±1.58 |
| FOMAL | 43.84±1.97 | 58.66±1.95 |
