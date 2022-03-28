# A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms

This is the repository associated to the paper:

**A Hitchhiker's Guide to Statistical Comparisons of Reinforcement Learning Algorithms.**

This code serves two purposes: reproducing the experiments from the paper and showing an example 
of rigorous testing and visualization of algorithm performances.

## Reproducing the Results:

### Running experiments:

`python3 run_experiment.py --study equal_dist_equal_var`

Possible studies:

* equal_dist_equal_var
* equal_dist_unequal_var
* unequal_dist_equal_var
* unequal_dist_unequal_var_1: here the first distribution is the one that has the smallest std
* unequal_dist_unequal_var_2: here the first distribution has the largest std

This creates a pickle file in ./data/equal_dist_equal_var/ for each pair of distributions.
A bash file is made available to launch the experiment on a slurm cluster.

It is advised to run the experiment with fewer iterations first, to make sure everything works.


### Plots and Tables

* To obtain plots of the false positive rates as a function of the sample size for various tests,
just run the plot_false_positive.py script:

    `python3 plot_false_positive.py --study equal_dist_equal_var`

* To obtain code for latex table that contains the statistical power results use the table_from_results.py script:

  `python3 table_from_results.py --study equal_dist_equal_var`


## Test and Plot two samples

`python3 example_test_and_plot.py`

The data we used are: 
* 192 runs of [Soft-Actor Critic](https://arxiv.org/abs/1801.01290) for 2M timesteps on [Half-Cheetah-v2](https://gym.openai.com/envs/HalfCheetah-v2/), using the [Spinning Up](https://github.com/openai/spinningup) implementation.
* 192 runs of [Twin-Delayed Deep Deterministic Policy Gradient](https://arxiv.org/abs/1802.09477) for 2M timesteps on [Half-Cheetah-v2](https://gym.openai.com/envs/HalfCheetah-v2/), using the [Spinning Up](https://github.com/openai/spinningup) implementation.

This example samples one sample of a given size from each, compares them using a statistical test and plot the learning curves, 
with error shades and dots indicating statistical significance.

The central tendency, the type of error, the test used, the confidence level and the sample size are tunable parameters.


## SAC and TD3 Performances

This repository also provides text files with learning curves for 192 runs of [Soft-Actor Critic](https://arxiv.org/abs/1801.01290) and 192 runs of [Twin-Delayed Deep Deterministic Policy Gradient](https://arxiv.org/abs/1802.09477)
run for 2M timesteps on [Half-Cheetah-v2](https://gym.openai.com/envs/HalfCheetah-v2/).
