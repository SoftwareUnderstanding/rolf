![license](https://img.shields.io/github/license/amaarquadri/perfect-information-game.svg)
![release](https://img.shields.io/github/v/release/amaarquadri/perfect-information-game?include_prereleases)
![total lines](https://img.shields.io/tokei/lines/github/amaarquadri/perfect-information-game)

# Perfect Information Game
Creating 2D perfect information board games, and playing them with machine learning systems.

## What's in this repo?
- Implementations for a bunch of board games including [Chess](https://en.wikipedia.org/wiki/Chess), 
  [Checkers](https://en.wikipedia.org/wiki/Draughts), [Connect 4](https://en.wikipedia.org/wiki/Connect_Four), 
  [Othello (Reversi)](https://en.wikipedia.org/wiki/Reversi), and [The Game of the Amazons](https://en.wikipedia.org/wiki/Game_of_the_Amazons)
- Implementations of various algorithms for selecting moves including [Minimax](https://youtu.be/l-hh51ncgDI) 
  (with alpha-beta pruning), [Monte Carlo tree search](https://youtu.be/UXW2yZndl7U) using randomized rollouts, 
  [iterative deepening](https://youtu.be/U4ogK0MIzqk?t=1566), and 
  [policy-value driven Monte Carlo tree search](https://arxiv.org/pdf/1905.13521.pdf)
- Code for generating [endgame tablebases](https://en.wikipedia.org/wiki/Endgame_tablebase) for Chess using 
  [retrograde analysis](https://www.chessprogramming.org/Retrograde_Analysis)
- Code for training neural networks from scratch using [self-play reinforcement learning](https://youtu.be/v9M2Ho9I9Qo)
- Over 4500 lines of Python code and counting
- Some cool [chess puzzles](https://www.github.com/amaarquadri/perfect-information-game/blob/master/training/KingOfTheHillChess/tablebases/README.md) that I discovered along the way

## Play Against Live Models on my Website
- Connect 4: [Easy](https://www.amaarquadri.com/play?game=connect4&difficulty=easy&ai-time=1&log-stats=true), 
  [Medium](https://www.amaarquadri.com/play?game=connect4&difficulty=medium&ai-time=1&log-stats=true), 
  [Hard](https://www.amaarquadri.com/play?game=connect4&difficulty=hard&ai-time=1&log-stats=true)
- Othello: Coming Soon
- Amazons (6x6 Board): Coming Soon

## Getting Started
- Ensure Python is installed
- Install requirements: \
`pip install -r requirements.txt`
- Play a game of Connect 4 against the ai: \
`python perfect_information_game/scripts/play_vs_ai.py`
- View games files that were generated during training: \
`python perfect_informationn_game/scripts/view_game_file.py`
- For games with multiple versions, select the desired version by opening the corresponding file under `src/games/` and 
uncommenting the corresponding line that starts with `CONFIG = `

## How I Trained the Models
- [Connect 4](https://www.github.com/amaarquadri/perfect-information-game/blob/master/training/Connect4/README.md)
- [Othello (Reversi)](https://www.github.com/amaarquadri/perfect-information-game/blob/master/training/Othello/README.md)
- [The Game of the Amazons (6x6 Board)](https://www.github.com/amaarquadri/perfect-information-game/blob/master/training/Amazons/6x6/README.md)

## Resources I Used
- [How to Keep Improving When You're Better Than Any Teacher - Iterated Distillation and Amplification](https://youtu.be/v9M2Ho9I9Qo)
- [Multiple Policy Value Monte Carlo Tree Search](https://arxiv.org/pdf/1905.13521.pdf)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
- [Introduction to Reinforcement Learning](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) (course by David Silver)
- [Parallel Monte-Carlo Tree Search](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.4373&rep=rep1&type=pdf)
- [A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm](https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf)
- [Time Management for Monte Carlo Tree Search](https://dke.maastrichtuniversity.nl/m.winands/documents/time_management_for_monte_carlo_tree_search.pdf)
- [Lessons From Alpha Zero (part 5): Performance Optimization](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)
- [How much training data do you need?](https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956)
- [Working with Numpy in Cython](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html)
- [Chess Piece Images](https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent)
- [Chess test cases](https://www.chessprogramming.org/Perft_Results) and more [chess test cases](https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9)
- [Chess Position Encoding Scheme](https://codegolf.stackexchange.com/a/19446)
