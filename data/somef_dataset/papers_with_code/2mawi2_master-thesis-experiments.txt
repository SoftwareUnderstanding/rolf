# master-thesis-experiments

This repository implements Proximal Policy Optimization. (Schulman et al., https://arxiv.org/pdf/1707.06347.pdf)

## Supported environments
OpenAI Gym:
```sh
"Hopper-v2", "Walker2d-v2", "FetchReach-v1", "InvertedDoublePendulum-v2"
``` 
## Setup

Setup your python3.7 virtual environment of choice.
I recommend [Anaconda](https://www.anaconda.com/).

Install all python dependencies.

If you use Anaconda, you can utilize the environment.yml file as follows:
```sh
$ conda env create -f environment.yml
```  
If you use virtualenv, you must install all dependencies manually.

Install [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) to evaluate experiments after running:

Acquire a valid [MuJoCo license](https://jupyter.readthedocs.io/en/latest/install.html) and install the MuJoCo version 2.0 binaries for Linux or OSX:

## Usage

### Get help:
```sh
$ python run.py --help
```  

### Run with default arguments:
```sh
$ python run.py
```  

### Evaluate a specific experiment with 6 different trials in parallel:

First, uncomment or modify the experiment of choice in "experiments.py".
By default, the experiment "test" runs a clipped PPO policy for 100000 steps on "Hopper-v2".
Editing the hyper-parameters of choice should be straightforward.

Then run the experiments file.
```sh
$ python experiments.py 
```  

Of you want to track the learning progress with [tensorboard](https://www.tensorflow.org/tensorboard/get_started) open a new bash and use:
```sh
$ tensorboard --logdir . --port 6999
```  
## Logs

Tensorboard directory:
```
data/tensorboard/DATE
```  

Logs of run.py
```
data/policy/logs/ENVIRONMENT/DATE
```

Logs of experiments.py
```
data/policy/results/EXPERIMENT/ENVIRONMENT/DATE
```

## Generate graphs from the logs:

Open graphs.ipynb with your [Jupyter Notebook App](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html).

Plot the experiment of choice.
Each plot will be generated from the last N runs of the directory: 
```
data/policy/results/EXPERIMENT/ENVIRONMENT/
```

Example graph for the "test" experiment for 6 evaluated runs on the Hopper-v2 environment:

```python
env = "Hopper-v2"
cats = [f"test"]
exp = "test"
labels = ["test"]
legend_title = 'Test Graph'

ax = plot_resampled(env,cats) # create resampled seaborn plot
ax.set_ylabel("Average Reward")
ax.set_xlabel(r"Timesteps ($\times 10^6$)")
ax.legend(title=legend_title, loc='lower right',fancybox=True, framealpha=0.5, labels=labels)
ax.set_xticklabels(['{:,.1f}'.format(x) for x in ax.get_xticks()/1_000_000])
ax.set_title(env)
figure = ax.get_figure() 
figure.savefig(f"{exp}-{env}.pdf", bbox_inches="tight") # save figure to "test-Hopper-v2.pdf"
```





 





