# Stateless Neural Meta-Learning with Second-Order Gradients

This is the Github repo associated with the paper: *Stateless Neural Meta-Learning with Second-Order Gradients*.
On a high-level, we propose a new technique TURTLE and compare its performance to that of transfer learning baselines, MAML, and the LSTM meta-learner in various challenging scenarios. In addition, we enhance the meta-learner LSTM by using raw gradients as meta-learner input and second-order information. 

## Techniques

All implemented techniques in this repository can be split into two groups:

- **Transfer Learning** (also used by [Chen et al. (2019)](https://arxiv.org/pdf/1904.04232.pdf))
  - Train From Scratch (TFS): Train on every task from scratch
  - Fine-Tuning (FT): Train a learner on minibatches without task structure, fix all hidden layers, and fine-tune the output layer on tasks
  - Centroid Fine-Tuning (CFT): Model with a special output layer. This layer learns vector representations for every class. Predictions are made by assigning the class with the most similar class representation to the input embedding. In similar fashion to the fine-tuning model, it is first trained on minibatches without task structure, and all hidden layers are frozen before performing meta-validation or meta-testing. 
  
- **Meta-Learning**
  - [LSTM Meta-Learner](https://openreview.net/pdf?id=rJY0-Kcll): Uses an LSTM to propose updates to the weights of a base-learner model. The code for this model comes from [Mark Dong](https://github.com/markdtw/meta-learning-lstm-pytorch) and has been adapted for our purposes.
  - [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/pdf/1703.03400.pdf): Learns a good set of initialization parameters for the base-learner. From this initialization, few gradient updates are enough to achieve good performance. Our MAML implementation uses 1 gradient update step and ignores second-order derivatives, as they were shown to be mostly irrelevant.
  - TURTLE (our proposed technique): A combination of the LSTM meta-learner and MAML where we replace the stateful LSTM by a stateless feed-forward network, and omit the default first-order assumption made by the LSTM meta-learner.

## Sine Wave Regression Experiments

The sine wave problem was originally formulated in [Finn et al. (2017)](https://arxiv.org/pdf/1703.03400.pdf). For our purposes, we have slighty the setup. That is, we do perform validation, even though it is not required (no fixed training set; tasks are only seen once so there is no risk of overfitting), as it gives us valuable information about the learning process. Second, we do not maintain a running average of performance over meta-training tasks. Instead, we use a fixed meta-test set, consisting of 2K tasks on which we evaluate the models' performances. 

The problem is as follows. Every task is associated with a sine wave function `f(x) = amplitude * sin(x + phase)`. The amplitude and phase are chosen uniformly at random for every task, from the intervals [0.1, 5.0] and [0, pi] respectively. Support sets contain k examples (x,y), whereas the query sets contain more than k observations to ensure proper evaluation. 

A base-learner neural network (input x -> dense layer (40) -> ReLU -> dense layer (40) -> ReLU -> output (1)) is used to learn the sine wave functions f. Thus, given a task, the goal is to infer the sine wave function that give rise to the examples from the support set. Correct inference leads to good performance on the query set. 

## Image Classification

Following [Chen et al. (2019)](https://arxiv.org/pdf/1904.04232.pdf), we implemented N-way, k-shot classification for the miniImageNet and [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) data sets. Both can be easily downloaded using a single script that we have created. More about this in the section about reproducability. 

In short, tasks are constructed by sampling N classes, and randomly picking k examples per class to construct the support set. Query sets are created by joining 16 randomly selected examples per class. The goal is to make a convolutional base-learner network learn as quickly as possible on the small support sets. The better it learns, the better the performance will be on the query sets. 

## Reproducability
We have made a special effort to make it easy to reproduce our results. Please follow the instructions below to do so. 

1. Clone this repository to your machine using `git clone <url>`, where `url` is the URL of this repo. 
2. Make sure you are running Python  3.6.10. Later versions should also work properly, please report it if this is not the case.
3. Install all required packages listed in *requirements.txt*. An easy way to do this is by using `pip install -r requirements.txt` We recommend you to create a virtual environment before doing this, using e.g., [miniconda](https://docs.conda.io/en/latest/miniconda.html). 
4. `cd` into the cloned Github repository and run *setup.py* using `python setup.py`. This will (i) download *miniImageNet* proposed by [Vinyals et al. (2016)](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) and process it using the splits proposed by [Ravi et al. (2017)](https://openreview.net/pdf?id=rJY0-Kcll), and (ii) it will download the [*CUB*](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset and prepares it for usage.  
5. Run *main.py* with the command `python main.py --arg1 value1 --arg2 value2 ...`, where *argi* are argument names and values the corresponding values. The script will try to load all parameters from the config file *configs.py* and overwrite these parameters with your provided arguments where necessary. 

### Reproducing the results from the paper

```python

## First-order MAML
python -u main.py --problem $1 --k_test 16 --model maml --validate --val_after 2500 --T 5 --model_spec FO-MAML-FINAL --k 1 --N 5 --cross_eval --T_test 10 --meta_batch_size 4 --runs 5

python -u main.py --problem $1 --k_test 16 --model maml --validate --val_after 2500 --T 5 --model_spec FO-MAML-FINAL --k 5 --N 5 --cross_eval --T_test 10 --meta_batch_size 2 --runs 5


## Second-order MAML
python -u main.py --problem $1 --k_test 16 --model maml --validate --val_after 2500 --T 5 --model_spec SO-MAML-FINAL --second_order --k 1 --N 5 --cross_eval --T_test 10 --meta_batch_size 4 --runs 5

python -u main.py --problem $1 --k_test 16 --model maml --validate --val_after 2500 --T 5 --model_spec SO-MAML-FINAL --second_order --k 5 --N 5 --cross_eval --T_test 10 --meta_batch_size 2 --runs 5

## First order LSTM meta-learner
python -u main.py --problem $1 --k_test 16 --validate --val_after 2500 --k 1 --N 5 --model lstm2 --model_spec FO-LSTM-FINAL --T 12 --runs 5 --cross_eval

python -u main.py --problem $1 --k_test 16 --validate --val_after 2500 --k 5 --N 5 --model lstm2 --model_spec FO-LSTM-FINAL --T 5 --runs 5 --cross_eval

## Second-order LSTM meta-learner
python -u main.py --problem $2 --k_test 16 --validate --val_after 2500 --k $1 --N 5 --model lstm2 --model_spec SO-LSTM-FINAL --T 8 --runs 5 --input_type raw_grads --second_order --cross_eval

## First-order TURTLE
python -u main.py --problem $2 --k_test 16 --k $1 --N 5 --model turtle --validate --val_after 2500 --lr 0.001 --model_spec FO-TURTLE-FINAL --layers 20,20,20,20,20,1 --input_type raw_grads --T 5 --meta_batch_size 2 --history grads --beta 0.9 --param_lr --runs 5 --cross_eval

## Second-order TURTLE
python -u main.py --problem $2 --k_test 16 --k $1 --N 5 --model turtle --validate --val_after 2500 --lr 0.001 --second_order --model_spec SO-TURTLE-FINAL --layers 20,20,20,20,20,1 --input_type raw_grads --T 5 --meta_batch_size 2 --history grads --beta 0.9 --param_lr --runs 5 --cross_eval


```

## Extending the code
We have tried to make it as easy as possible to extend our code. We have created high-level abstractions for core objects. Below you find instructions to create your custom algorithm or data loader. 

### Creating your custom algorithm
To create your own algorithm, you can create a new file called `youralgorithm.py` in the *algorithms* folder. In that file, you can define your own algorithm class which inherits from the *Algorithm* class specified in *algorithm.py*. You will need to implement 3 functions: __init__ (the initialization function), train (train on a given task), and evaluate (apply your algorithm to the task and obtain the performance on the query set). You can use other defind algorithms as examples on how to do this. 

Once you have done this, you can define default arguments for the __init__ function in *configs.py*. Lastly, you can import your config, add a string identifier for your algorithm to the `choices` field in the --model argument in *main.py*, and add it to the dictionary `mod_to_conf`. You are then fully set to run your new algorithm by calling `main.py` with the argument `--model youralgorithmspecifier`!

### Creating your own data loder
For data loaders, we have also defined an abstraction class *DataLoader* in *data_loader.py*. To create your own data loder, make sure to download the required files first. Afterwards, you can create your own file `yourdataloader.py` and define a class that inherits from `DataLoader`. Your class should have at least 4 functions: __init__ (initialization, where you load the data from files), _sample_batch (sample a flat batch of data for baseline models), _sample_episode (sample a task), and generator (generator object that iteratively yields batches or episodes depending on the provided argument). You can follow the examples of `sine_loader.py` and `image_loader.py`. 

Once you have created your own DataLoader class, you should import it into the *main.py* file, add an option to the --problem argument for your new problem, and add an `if args.problem == yourproblem` statement to the `setup` function. You are then ready to run the algorithms on the new data set by simply running `main.py` with the argument `--problem yourproblem`!


