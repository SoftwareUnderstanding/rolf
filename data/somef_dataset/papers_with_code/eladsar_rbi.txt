# rbi
Implementation of distributed RL algorithms:

Baselines:
1. Ape-X (https://arxiv.org/abs/1803.00933)
2. R2D2 (https://openreview.net/pdf?id=r1lyTjAqYX)
3. PPO (https://arxiv.org/abs/1707.06347)

RBI:
A safe reinforcement learning algorithm 

Currently, supported environment is ALE
## How to run

A distributed RL agent is composed of a single learning process and multiple actor process.
Therefore, we need to execute two bash scripts one for the learner and one for the multiple actors.

choose \<algorithm> as one of rbi|ape|ppo|r2d2|rbi_rnn 

### Run Learner:

sh learner.sh \<algorithm> \<identifier> \<game> \<new|resume>

resume is a number of experiment to resume.
For example:

```bash
sh learner.sh rbi qbert_debug qbert new
```

starts a new experiment, while:

```bash
sh learner.sh rbi qbert_debug qbert 3
```

resumes experiment 3 with identifier qbert_debug

### Run Actors:

sh actors.sh \<algorithm> \<identifier> \<game> \<resume>

### Run Evaluation player:

right now there are two evaluation players in each actors script

### Terminate a live run:

1. ctrl-c from the learner process terminal
2. pkill -f "main.py"  (kills all the live actor processes)
3. rm -r /dev/shm/<your name>/rbi/* (clear the ramdisk filesystem)

### Setup prerequisites before running the code

#### To login: 
ssh \<username>@\<server-address>

Use ssh-keygen and ssh-copy-id to avoid passwords:
```bash
ssh-keygen
ssh-copy-id -i ~/.ssh/id_rsa user@host
```

#### Install Anaconda:
copy anaconda file to server and run:
sh Anaconda3-2018.12-Linux-x86_64.sh

#### Install Tmux:
make new directory called tmux_tmp
copy ncurses.tar.gz and tmx-2.5.tar.gz to tmux_tmp directory
copy install_tmux.sh to server and run
./install_tmux.sh

setup directories, clone rbi and setup conda environment

```bash
mkdir -p ~/data/rbi/results
mkdir -p ~/data/rbi/logs
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/eladsar/rbi.git
cd ~/projects/rbi
conda env create -f install/environment.yml
source activate torch1
pip install atari-py
```

### Docker

We also provide a docker file and instructions to build and run the simulation in a docker container.

Please first, install nvidia-docker:

https://github.com/NVIDIA/nvidia-docker

To build the docker container:

```bash
cd install
docker image build --tag local:rbi .
```

To run the docker container:
```bash
nvidia-docker run --rm -it --net=host --ipc=host --name rbi1 local:rbi bash
```

## Evaluation

There are three ways to evaluate the learning progress and agent performance

### Tensorboard

Each run logs several evaluation metrics such as: 
(1) loss function
(2) network weights
(3) score statistics (mean, std, min, max) 

To view the tensorboar run an ssh port-forwarding command

```bash
ssh -L <port>:127.0.0.1:<port> <server>
```
and from the server terminal run
```bash
cd <outputdir>/results
tensorboard --logdir:<name>:<run directory> --port<port>
```

### Jupyter Notebook

To view a live agent run the evaluate.ipynb notebook.
Use the identifier name and the resume parameter to choose the required run.
You may also need to change the basedir parameter.

[![A visualization of a Qbert RBI agent](https://img.youtube.com/vi/pWHybdalu1g/0.jpg)](https://www.youtube.com/watch?v=pWHybdalu1g)

### Pandas Dataframe

Performance logs are stored to numpy files and in the end of the run a postprocessing process stores all logs into a pandas dataframe.
These dataframes may be used to plot the performance graph with the plot_results.py script.
