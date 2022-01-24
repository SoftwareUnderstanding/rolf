#### Recurrent Dropout without Memory Loss

Theano code for the Penn Treebank language model and Temporal Order experiments in the paper [Recurrent Dropout without Memory Loss](https://arxiv.org/abs/1603.05118).

#### Requirements

**Theano** is required for running the experiments:

```bash
pip install Theano
```

#### Language Modeling Experiments

First, run makedata.sh. Then select model to run in config.py and run main.py.

To run the baseline models:
```bash
python -u main.py
```

To run the models with 0.25 per-step dropout in hidden states:
```bash
python -u main.py --hid_dropout_rate 0.25 --per_step
```

To run the models with 0.5 per-sequence dropout in hidden state updates:
```bash
python -u main.py --hid_dropout_rate 0.5 --drop_candidates
```

#### Temporal Order Experiments

To run the baseline LSTM without dropout:
```bash
python -u order.py
```

To run the models with 0.5 per-step dropout in hidden state updates on 30 symbol long sequences:
```bash
python -u order.py --hid_dropout_rate 0.5 --drop_candidates --per_step --low 10 --high 20 --length 30
```
