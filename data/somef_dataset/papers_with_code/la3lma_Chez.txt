# chezjulia


A little "fingering exercise" to get a feel for Julia, may also contain
elements of a chess program ;-)

To run it, start julia, then run

      include("src/Chez.jl")

This will load the code and run unit tests. 


## TODO


* Make instructions for how to start easily from the command line.

### Bugfixes

* Make playing consistently work again

* Write games (or at least some of them) to a log so they can be inspected by
  a human to see where the machine get stuck in loops etc.


* Persist player state is buggy in several ways.
  * Inconsistent states can be written (fix by renamming
    file after correctely writing it).
  * The representations are being bloated, for some reason they seem to
    grow in size in a way that is totally out of proportion to
    the amount of information that is in the weigh chain.

* Log the average/median  length of non-draw games.  Use it to
  track progress in learning (from a hypothesis that better players
  will win or lose in shorter games).

* Tweak Q metric to give a bias to all moves in a winning game
  regardless of length.

* Make interface to VS work nicer.

* Make micro game that plays really fast to debug the
  persistence bug.

* Make the interface to the loss function much less baroque.

* The whole package/module thing is still a little confusing to me,
  so there is probably something wrong there.

* Once the basics (see above) are fixed, then fix these two
  serious bugs:

  * The chain architecture is just a placeholder, fix it to be useful.
  * The loss function needs a very thorough review, it's almost certainly
    wrong.
    

### Optimizations

* Make it possible to run this thing, with snapshotting, for weeks.
* Make the flux calculations run using a GPU
* Make the scalar computations (chessplaying in particular) run in
  paralell (one process per game). It's a rediculously parallell
  task, and I want to speed it up by a factor of how many cores are
  available.

### Game mechanics

* Detect move repetition to signal draws.
* Let the game board contain the state of the rookings, and movements of the king/rooks.
* Implement en-passant
 

### Reflection on the current state

I do have a halfway decent implementation of the game mechanics of
chess. It doesn't have rookings, en passant or draw by repetition
implemented, but that can be added by spending time, and it will not
change any external interfaces.  I'm not giving that much priority
now.  Efficiency can obviously be improved, but I'm also not giving
that any priority now.

What I don't have at all is a functioning reinforcement learning
setup, so I need to work on that.  Chess is a complicated game with
high move fan-out that takes some time to play so it may be worth it
to experiment with the reinforcement learning strategies on a simpler
game, but to make sure that the chess game can be plugged in an run on
it at any time (keep that as an unit test)

There are a few chess-specific pieces of code in the reinforecement
learning thing, and weeding those out will in themselves be useful.

Now, using a simpler game, like "four in a row" will also make it
possible to learn from other people's work (see some references
below).  That will obviously be useful too.


The actual Q-learning, SARSA, position/move evaluation whatever
algorithms that are plugged in, with whatever generic heuristics
can be added, needs to be _crystal_clear_.   The current
implementation of "q-learning" (which has already mutated)
is quickly getting messy.  That cannot be permitted to happen.
The core algorithm must be crystal clear for the reader, and the
rest of the supporting software must enable this clarity.

Finally, but not least importantly: I need to add instrumentation to
track progress.  It is nice to get little printouts as we move along,
but it is also important to log performance over time in an orderly
consistent manner, and to be able to plot evolution of performance
over time.   Some ideas are:

 * Use sqlite or the julia-native database thing to log values.
 * Re-read the Q-learning papers (and other) and shamelessly copy
   their metrics and graphs.  Reproduce them.
 * Use these metrics to track progress of the learning algorithms
   over time, across classes of games, and instances of games.
 * Don't be cute, use denormalized tables, one per metric,


### Running on GPU with Julia

* Take a look at  docker run -it --rm --gpus all nvcr.io/hpc/julia:v1.2.0 /workspace/examples/test_cudanative.jl, an see
  if that image is usable out of the box.  If it is, that's nice.

* Can take a look at this https://github.com/maleadt/julia-ngc/blob/master/Dockerfile

* These flags are necessary -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility

* https://github.com/JuliaGPU/CUDA.jl/commit/fa32bca8f97b764d5d83fefc1bea0d4282e86d0d


### Strategy development

* Deep neural network, reinforcement learning player based on Flux.jl
   - Map game-states into neural network representations
   - Design a network architecture to either find next move or evaluate positions.
   - Design a "goodness" criterion for strategies.


References
### 
* https://github.com/tensorflow/minigo
* https://arxiv.org/pdf/2001.09318.pdf
* https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/
* https://web.stanford.edu/~surag/posts/alphazero.html
* https://theaisummer.com/Deep-Learning-Algorithms/


### Deep q learning

* Dueling network architectures for reinforcement learing: https://arxiv.org/pdf/1511.06581.pdf
* Dueling deep q networks: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
* Distributed deep q learning: https://stanford.edu/~rezab/classes/cme323/S15/projects/deep_Qlearning_report.pdf


### Four in a row

* https://medium.com/@sleepsonthefloor/azfour-a-connect-four-webapp-powered-by-the-alphazero-algorithm-d0c82d6f3ae9
* https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a
* https://makerspace.aisingapore.org/2019/04/from-scratch-implementation-of-alphazero-for-connect4/
* https://timmccloud.net/blog-alphafour-understanding-googles-alphazero-with-connect-4/


### Chessboards


Consider using http://chessboardjs.com/docs to add som real chessboard action.

this would be relatively easy, if I first implement a printer for
Forsyth-Edwards notation to encode board positions
(https://en.wikipedia.org/wiki/Forsyth–Edwards_Notation), since
chessboardjs is capable of decoding FEN encoded board positions.


Using the FEN viewser  from the chrome market may be sufficient: https://chrome.google.com/webstore/detail/simplechessboard/hppnfmeaoiochhjdlojgflkfedncdokl

... just pasting the game into this webpage will make it possible to view it, and to analyse it.

Also output PGN notation to allow analysis using external tooling

https://en.wikipedia.org/wiki/Portable_Game_Notation