## LunarLander Approximate Q-Learning

#### What is this project?
This is an attempt to solve the LunarLander-v2 ENV defined in OpenGym AI, using Deep Q-Learning. We create a double 
layered neural network with 8 observation vectors and 4 possible actions in each state. We use Approximate Q-Learning
to create an optimal solution to allow the Lunar Satellite to land gently in the marked helipad to maximize rewards. We
compare between two Keras optimizers: Stochastic Gradient Descent 
(https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L164) and Adam Stochastic Optimization. 
(https://arxiv.org/abs/1412.6980v8)

#### How to get started with this project?
To install our python dependancies in this project we are using pipenv.
In order to get started with pipenv follow the installation guide at https://github.com/pypa/pipenv

You also need graphviz to view data-visualizations. You can install it by following the installation guide at https://graphviz.gitlab.io/download/

Once you have pipenv installed run:
```python
pipenv install
pipenv run python3 q_learning.py
```

#### File Contents
Approximate Q-Learning with Adam optimization for 10 epochs and 500 games
```
Adam_10ep_model_accuracy.png - Epoch v/s Accuracy graph
Adam_10ep_model_loss.png - Epoch v/s Loss graph
LunarLander-v2-Adam-10ep-weights.h5 - Weights h5 file storing weights for each feature for the trained model
```
Approximate Q-Learning with Adam optimization for 20 epochs and 500 games
```
Adam_model_accuracy.png - Epoch v/s Accuracy graph
Adam_model_loss.png - Epoch v/s Loss graph
LunarLander-v2-Adam-weights.h5 - Weights h5 file storing weights for each feature for the trained model
```
Approximate Q-Learning with SGD optimization for 10 epochs and 500 games
```
SGD_10ep_model_accuracy.png - Epoch v/s Accuracy graph
SGD_10ep_model_loss.png - Epoch v/s Loss graph
LunarLander-v2-SGD-10ep-weights.h5 - Weights h5 file storing weights for each feature for the trained model
```
Approximate Q-Learning with SGD optimization for 20 epochs and 500 games
```
SGD_model_accuracy.png - Epoch v/s Accuracy graph
SGD_model_loss.png - Epoch v/s Loss graph
LunarLander-v2-SGD-weights.h5 - Weights h5 file storing weights for each feature for the trained model
```
