# Adversarial Soft Advantage Fitting - 1 (ASAF-1)
Imitation learning without policy optimization !

## Introduction (In Traditional Chinese):
https://www.notion.so/Adversarial-Soft-Advantage-Fitting-441698eb0ccb40eab4f59275d637466a

## Description

### Preparing expert demos:
Every expert demo (state-action pairs) file must be a pickle file and in this form: [[np.array([state0]), np.array([action0])], [np.array([state1]), np.array([action1])], ...]

### Training:
Adjust the parameters in ./src/train.py and then run it.

### Testing:
Adjust the parameters in ./src/test.py and then run it.

## Experiment
![](./image/SS%201.png)  

## Reference
https://arxiv.org/abs/2006.13258
