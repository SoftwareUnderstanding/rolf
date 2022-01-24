# alphazero-clone

This is meant to serve as a clone of the AlphaZero architecture, discussed [here](https://arxiv.org/abs/1712.01815). The architecture is reproduced in both keras and pytorch.

There are still a number of bugs to resolve before results can be generated.

# Dependancies

* numpy
* keras
* pytorch
* python-chess (when using the chess game)

# Usage

The `main.py` file contains an example of setting up a training session for the AlphaZero architecture. One can select the backend (pytorch or keras) by changing the `nnet.keras` line to `nnet.pytorch`. You can play against the trained model by running `play.py`.

Three games are available, being either TicTacToe, Othello, or Chess. TicTacToe and Othello board sizes can be changed by utilising their parameter setting, e.g. `TicTacToe(5)` produces a 5x5 TicTacToe board, and `Othello(10)` would produce a 10x10 Othello board.
