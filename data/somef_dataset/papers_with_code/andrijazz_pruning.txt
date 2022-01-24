# Pruning

Analysing pruning strategies.

### Dependencies

I am using [pipenv](https://pipenv-es.readthedocs.io/es/stable/) and [W&B](https://www.wandb.com/) for visualizations.

### Setup

```
# checkout the code and install all dependencies
git clone https://github.com/andrijazz/pruning
cd pruning
bash init_env.sh <PROJECT_NAME> <PATH_TO_YOUR_DATA_STORAGE>
pipenv install

# activate venv
pipenv shell

# train model
python run.py --mode train

# test models and plot results to w&b
python run.py --mode test

```

### Observations

![](https://github.com/andrijazz/pruning/blob/master/docs/plot2.png)

I was expecting that accuracy will start dropping much sooner. Pruning more then 90% of weights with almost none degradation in performance really surprised me. Weight pruning preforms better because pruning the unit discards the portion of information by propagating zeros to the next layer while this is not the case with weight pruning.

### TODOs
* More experiments - check the results on more complicated networks
* How can we use this for interpretability?
* Implement pytorch SparseLinear module using https://pytorch.org/docs/stable/sparse.html experimental API and measure performance gain
* TF implementation

### References
* https://jacobgil.github.io/deeplearning/pruning-deep-learning
* https://for.ai/blog/targeted-dropout/
* https://arxiv.org/pdf/1905.13678.pdf
