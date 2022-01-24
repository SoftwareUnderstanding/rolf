# vamperouge

## Usage

`python3 vamperouge.py {adresseIP} {port}`

`pytorch` and `numpy` are required.

## Files

Related to Alpha Zero

- MCTS implemented in `mcts.py`
- DNN implemented in `model.py`
- self-play implemented in `self_play.py` and `arena.py`

Related to Vampires VS Werewolves

- game logic implemented in `game.py`
- TCP client implemented in `tcp_client.py`

Run `python3 main.py` to start training Vamperouge (don't run this on your machine, it uses a LOT of RAM).

## Credits

This implementation of the Alpha Zero algorithm was inspired by:
- https://github.com/suragnair/alpha-zero-general (main reference)
- http://web.stanford.edu/~surag/posts/alphazero.html
- https://arxiv.org/pdf/1712.01815.pdf
- https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
- https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191
- https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

Also, for game-specific things, credits to:
- https://github.com/Succo/twilight
