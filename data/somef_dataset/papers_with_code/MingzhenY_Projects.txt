# Projects

## Awari

After reading the AlphaZero paper, the idea that the program learns only by selfplay is intresting, the only inputs needed are the game rules.
The code is written using template so that it's easy to define a new game
and play against computer.
Might add methods other than MCTS in the future.  

To use the code, define your own game class in games.h.
See main.cpp for examples.

games.h:  
The game classes are defined here.
It should be a two-player perfect information game.

gametest.h:  
Given the game class, checks if it is well defined.

mcts.h:  
Uses Monte-Carlo Tree Search to find best moves.

gameplay.h:  
The code to play the game against computer.

main.h:  
main code

References:  
1.[AlphaZero(arXiv:1712.01815)](https://arxiv.org/abs/1712.01815)  
2.Allis, L. V. Searching for Solutions in Games and Artificial Intelligence.  
PhD thesis, Univ. Limburg, Maastricht, The Netherlands (1994).  
3.Russ, Larry. The complete mancala games book.  