# rl18-curiosity

Student research project in a [course](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/contenus-/reinforcement-learning-214281.kjsp?RH=1242430202531) of Reinforcement Learning.

**WORK IN PROGRESS**

Our goal here is too study some recent advances in curiosity-driven exploration: our contributions should be both experiments and a research report with [NeurIPS style](https://nips.cc/Conferences/2018/PaperInformation/StyleFiles).

## Project description

Environments where the extrinsic rewards are sparsely observed are harder to explore and to learn for RL agents since they may require more episodes to determine which sequences of actions lead to high rewards. Recently, it has been argued that adding intrinsic rewards (for instance through Random Network Distillation) can help to drive the exploration efficiently in Atari games [1][2][3] and can be simply implemented along with Proximal Policy Optimization [4]. As opposed to extrinsic rewards, intrinsic rewards are not necessarily constant over the episodes: such rewards can be seen as "curiosity" functions. Through this case study we aim at highlighting the advantages and limitations of curiosity-driven exploration.

[1] https://arxiv.org/pdf/1810.12894.pdf
[2] https://arxiv.org/pdf/1705.05363.pdf
[3] https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf
[4] https://arxiv.org/pdf/1707.06347.pdf

## Contents

* [reviews](./reviews)
* [models](./models)

## Requirements

This repository is developed on Linux 64-bit in Python 3.7 using [(Mini)conda](https://conda.io/miniconda.html).

To instanciate the conda environment, run ``conda create --file conda-env.txt --name rl``.

Then use pip to install the required Python packages: ``pip install -r requirements.txt``.

To work on Jupyter notebook or Jupyter lab you will need to install it via pip. Follow the instructions [here](https://anbasile.github.io/programming/2017/06/25/jupyter-venv/) to make the environment kernel available in your jupyter installation. Suggested command: ``ipython kernel install --user --name=rl``
