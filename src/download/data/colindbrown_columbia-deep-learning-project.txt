# Deep Learning Final Project

**Colin Brown**
**Jayant Subramanian**
**Tanmay Chopra**

This is the repository for our COMS4995 Deep Learning Final Project, in which we investigate different methods of finding approximations to the NP-complete minimum vertex cover problem.

This code is written using python 3.7 and PyTorch 1.4.0.
Our requirements are in **requirements.txt**, to install run `pip install -r requirements.txt` in a new virtual environment.

## MCTS
We first use Monte-Carlo tree search to geta baseline for approximating the Min Vertex Cover problem, as well as comparing different formulations of the game to see their relative performance.

To view and run this code, run `jupyter notebook` and navigate to **MCTS/mcts.ipynb**.

## AlphaZero
We then implemented a the AlphaZero algorithm to approximate the MCTS policy of which vertex to add to the cover next, given a particular graph and existing cover. We used the **torch-geometric** package to implement the neural network using Graph Convolutional Layers.

To view and run this code, run `jupyter notebook` and navigate to **AlphaZero/alphazero.ipynb**.

References:
* https://arxiv.org/abs/1905.11623
* https://arxiv.org/abs/1712.01815

## MuZero
We finally adapted an implementation of the MuZero algorithm to compare the performance of its fully learned model. Our implementation is adapted from [this repository](https://github.com/johan-gras/MuZero) by Johan Gras, with the entire respository converted from using Tensorflow to PyTorch for compatibility with torch-geometric.

To run the code, run `python MuZero/muzero.py`

Notes about implementation:
* We use torch-geometric GraphSage layers for the representation network
* Our dynamics and prediction functions are each split into two networks, so that each network is only predicting one quantity
* We also update the loss functions from the original paper.

References:
* https://arxiv.org/abs/1911.08265
