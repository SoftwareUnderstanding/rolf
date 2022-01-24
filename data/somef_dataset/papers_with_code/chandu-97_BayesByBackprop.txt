# Bayes by backprop

This repo consists of basic version of Bayes by backprop(MLP). Deep neural networks have achieved impressive results on a wide variety of tasks. However, they are prone to overfitting. When used for supervised learning, they tend to make overly confident decisions about the output class and don't estimate uncertainty of the prediction. On the other hand, Bayesian Neural Networks  can learn a distribution over weights and can estimate the uncertainty associated with the outputs.

## Background Papers
These are the background papers used:-

https://arxiv.org/abs/1505.05424

For a clearer understanding of Variational Free Energy loss you can refer to:-

https://www.cs.toronto.edu/~graves/nips_2011.pdf

https://arxiv.org/abs/1312.6114

## Output

![Example output for MNIST](Example.png)

You can look at abalation runs [here](https://app.wandb.ai/chandu/Week%205%20BBB%20Abalations?workspace=user-chandu)

## Getting Started
I would suggest to use a seperate virtual environment for this purpose so that it doesnot conflict.

### Virtual environment setup

```
virtualenv --python=python3.6 bbb
source bbb/bin/activate
```

### Installation steps
```
git clone --depth=1 https://github.com/luke-97/BayesByBackprop.git
pip install -r requirements.txt
```

## Running
If you are using Wandb use the key to login
```
wandb login <key(in wandb website)>
```
```
cd bbb
python test.py

```
Need to implement for Convs

Special thanks:-

1)https://github.com/kumar-shridhar/PyTorch-BayesianCNN

2)https://gist.github.com/vvanirudh/9e30b2f908e801da1bd789f4ce3e7aac
