# dropout_privacy
Final project for CS590.

This project seeks to explore the relationship between dropout as an uncertainty measure as explored in the work of Yarin Gal and Zoubin Ghahramani. [link](https://arxiv.org/abs/1506.02142)
We run a membership inference attack against trained models and assess how Dropout and Differential Privacy interoperate to protect training set data.

## Preliminaries
In order to run the experiment, install the reqiured libraries.

## Running the experiment
1. Run `model.py` to train the required target models.
2. Run `attack.py` to attack the models.

Note that it is easy to monitor performance of the models during training and inference time by using [Tensorboard](https://www.tensorflow.org/tensorboard)
