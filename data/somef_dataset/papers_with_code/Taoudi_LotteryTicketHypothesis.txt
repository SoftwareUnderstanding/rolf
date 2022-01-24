# LotteryTicketHypothesis
By Kevin Ammouri and Youssef Taoudi


Project aiming to replicate The Lottery Ticket Hypothesis by J. Frankle and M. Carbin (https://arxiv.org/abs/1803.03635)

Winning tickets were examined using the LeNet-300-100(LeCun) architecture on the MNIST dataset and miniature versions of the VGG model(Simonyan,http://arxiv.org/abs/1409.1556) for CIFAR-10.




conv_cifar consists all code, data and plots for the convolutional network pruning while fc_mnist contain all code, data and graphs for the fully connected network pruning.

##### fc_mnist
```constants.py``` - architecture and hyperparameter setup

```experiment.py``` code for running the LeNet experiments

```conv_models.py``` - code for creating and training the models as well as applying the mask for pruning

```plots.py``` - code for plotting

```pruning.py``` - code for pruning

```tools.py``` - helper functions

##### conv_cifar
```constants.py``` - architecture and hyperparameter setup

```conv_experiment.py``` - code for running the ConvNet experiments

```conv_models.py``` - code for creating and training the models as well as applying the mask for pruning

```plots.py``` - code for plotting

```pruning.py``` - code for pruning

```tools.py``` - helper functions
