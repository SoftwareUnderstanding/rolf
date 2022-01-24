# ThinkingTicTacToe
An attempt to create a Tic Tac Toe player which learns by self play inspired by Alpha Zero.

Read more about Alpha Zero and the algorithm:
https://arxiv.org/pdf/1712.01815.pdf
https://deepmind.com/documents/119/agz_unformatted_nature.pdf

Much inspiration follows thanks to Jeff Bradberry
https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/

The Project development consists of three stages:

Stage 1: Using UCB algorithm in monte carlo search tree to pick the strongest play.
Please download all the files in UCB folder. Run ThinkingTicTacToe.py to play against the UCB Player in a console.

Stage 2: Using the best moves from UCB algorithm to train a neural network which predicts best move.
Please download all the files in SL folder. Run alphaZeroTrain_SL.py to train the neural network. You will need keras and tensorflow for this. The number of games to be trained for are specified in variable called gamesTrainBatch. Once training is complete, you can run playWithAlphaZero.py to play against the trained network.

Stage 3: Using AlphaZero algorithm to do reinforcement learning
Under progress
