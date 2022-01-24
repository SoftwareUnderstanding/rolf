# Chess AI

Developed by Ryan Pope for EECS 649 (Intro to Artificial Intelligence).

## Overview

Here is a link to a youtube video explaining my code: https://www.youtube.com/watch?v=M9VLWW0a1_4 

This repo contains several Jupyter notebooks to explore chess algorithms using python. I utilized the python library python-chess in order to execute and manage all of the chess backend features. I began with simple random move generation and slowly upgraded to using a minimax algorithm with basic material evaluation to determine best moves,

After researching about AlphaZero and Leela Chess Zero, I tried to implement some similar approachs utilizing UCT Monte Carlo Tree Search to generate move sequences and a neural network that was trained to evaluate different positions using previous stockfish evaluations from the lichess dataset.

## Data 

I downloaded the May 2015 Lichess dataset from https://database.lichess.org . I then filtered games by Elo of both players greater than 1800 to generate "better" play and I used only games with the stockfish evaluations. I iterated over the moves of the game and created frames from the PGNs where I utilized an array of size 64 to store the encoded board. 

## Files

`filtered_games.pgn` is the filtered PGN dataset that I utilized for training.

`preprocessed.csv` is a CSV file containing the data as arrays of 64 values plus the evaluation for the state.

`ChessExploration.ipynb` is a Jupyter notebook with basic algorithms such as Minimax.

`Preprocessing.ipynb` converted the PGN dataset into the arrays stored in the csv.

`NeuralNetTraining.ipynb` trained the neural network.

`MCTSChess.ipynb` implements the Monte Carlo Tree Search and utilizes the Neural Net to evaluate positions.

## Depenedencies

* Keras
* Tensorflow
* Numpy
* Scikit-learn
* python-chess

## Key Resources

Papers:

* https://arxiv.org/pdf/1712.01815.pdf
* https://arxiv.org/pdf/1509.01549.pdf

Articles:

* http://stanford.edu/~cpiech/cs221/apps/deepBlue.html
* https://medium.freecodecamp.org/simple-chess-ai-step-by-step-1d55a9266977

Code Influences:

* https://chessprogramming.org
* https://github.com/niklasf/python-chess
* https://python-chess.readthedocs.io/en/latest/index.html
* http://mcts.ai/code/python.html
* https://jupyter.brynmawr.edu/services/public/dblank/CS371%20Cognitive%20Science/2016-Fall/Programming%20a%20Chess%20Player.ipynb
* https://github.com/leela-zero/leela-zero/

Other:

* https://lczero.org
