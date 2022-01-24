# Overview

This repository contains an implementation of the paper [Proximal Policy Gradients](https://arxiv.org/abs/1707.06347).

# Required packages
- ``gym`` for environment. Install MuJoCo if you want to run FetchPickPlace-v1
- ``termcolor`` for colored output. *Not needed yet, logging is in progress*
- ``scipy`` for saving ``.mat`` files. *Not needed yet, logging is in progress*
- ``matplotlib`` for plotting graphs
- ``numpy`` and ``pytorch`` for the algorithm

# How to run the program
Create the folder ``log/``. This is required to store logs.

The main program is located in ``src/main.py``. To look at help for the
command, run ``python main.py -h``. An environment is required to run the
program.

The plotting program is located in ``src/plot.py``. To plot the training
statistics after the program has halted, run ``python plot.py path/to/logdir``.
The logging directores are located in ``log/datetime`` where the numbers are of
the form ``year-month-date-hour-minute-second``.

## Example
``python main.py --ip --discount 0.75 --data-iterations 10 --value-iterations 15 --policy-iterations 150 --log --epochs 200 --max-kl 150``
