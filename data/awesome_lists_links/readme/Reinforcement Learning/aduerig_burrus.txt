# othello
## Introduction
Burrus is an AI programmed to learn how to play Othello. Othello is a two-player, perfect information game that is also known as Reversi. Read more about the game here:
https://en.wikipedia.org/wiki/Reversi

Burrus uses reinforcement learning with a neural network to learn how to play Othello from scratch. The core learning algorithm is a re-implementation of the novel algorithm introduced in the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" published by Google DeepMind: https://arxiv.org/abs/1712.01815. A simple version of the learning loop is as follows:


1. Initialize neural network with random weights 
1. Self-play the network against itself with random noise added for diversity in moves
1. Train next iteration of the neural network with the winner of games from step 2.
1. Repeat steps 2-4.


Burrus comes with an extremely performant Othello engine to compute legal moves and gamestates efficiently. Bitboard calculations are adapted and applied to this new game domain from the modern literature on writing performant chess engines (for example of performant chess engine, see: Stockfish https://github.com/official-stockfish/Stockfish)


## Running Burrus
### Requirements to build and run
* Unix based environment
* Python 3.5 or above
	* numpy 
	* tensorflow 
	* posix_ipc
* std c++14

### Training burrus
If training on one computer, first run `make`

then run `train_serial.sh`

### Playing against burrus
Running `play` after `make` will have two players as defined in `play.cpp` play against each other.

#### Types of players avaliable
* Human: Accepts input from the keyboard for moves to make. Takes in an integer from 0-63 starting in the upper left and going right.
* Rand: Makes random legal moves.
* Minimax: Traditional AI that looks ahead to a certain `depth` game states, and minimizes the loss from its perspective according to a simple `board_evaluation` function (total_enemy_pieces - total_its_pieces). Depth can be passed in as an argument to the script `play` 
* MonteCarlo: AI that uses the AlphaZero rollout algorithm to select its next move.


## Help running on bridges

Bridges is a supercomputing cluster that was used during the creation and training of burrus:
https://www.psc.edu/bridges
Bridges allowed burrus to execute hundreds of games in parellel, all with seperate GPUs.


To run interactively (not with a job script):  
	- Run "module load tensorflow/1.5_gpu gcc"  
	- Run "pip install --user posix_ipc"  
	- Then run "interact -gpu" to get a hold of gpu  
	- Run "make"  
	- The bridges environment should be ready to run any of the files outputted from make. 

Warning, if "module purge" is run, then "module load slurm/default psc_path/1.1" must be run to interact with the GPU.

If you encounter an error that tensorflow session is unable to be created, it means that your GPU memory is full.

Run "nvidia-smi" to check on gpu memory usage.

If you encounter the error: "Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms)", then delete the .nv folder from your home directory


## Description of files

### cnn_resnet.py
The main model for the MonteCarlo players to use. Uses tensorflow. Contains the train() script that play_and_train.py calls.

### param_serial.cpp
Plays a certain number of games on one processor. Command line args are -rank and -ngames. Spawns a python_model_communicator.py to comminucate with.  
Example call:  
./param_serial -rank 0 -ngames 10

### python_model_communicator.py
Runs from param_serial.cpp and communicates through shared memory. Serves requests for tensorflow forward passes.

### play_and_train.py
Calls the hi_mpi_script.sh and cnn_train.sh over and over again performing the full improvement loop.

### engine.hpp + engine.cpp
The engine for the othello game. Handles playing moves, finding legal moves, and tracking the board state.

### player.hpp + player.cpp
Definitions for types of players.

### driver.cpp
Plays two montecarlo players against each other and reports the win.

### play.cpp
Plays two players against each other and reports the win with timing results.
