# MNIST Sample Models

This repo contains (a) quick implementation(s) of digit recognition from the MNIST dataset.
This is repo is a more accurate representation of my coding ability after my internship at Pinterest :)

## Model Plans
1. ~~embedding based basic model~~ (try something different) <br/>
 a. ~~Set up tensorboard logging~~
 
2. ~~Auto-encoder~~ (most guides of this don't show actual implementation. Will implement based on the basic model as
multi-task learning)

3. ~~Mixture of Softmaxes~~ (This shouldn't help but just for fun I'll implement) <br/>
    (https://arxiv.org/pdf/1711.03953.pdf)
    
4. ~~CNN~~ (Very typical to be used for this problem)


## Overview
This repo contains a few models used to classify hand-written digits from the MNIST dataset as well as their performance
metrics. 

## Required Packages
Python 3.6.3

tensorflow==1.10.0 <br/>
tensorboard==1.10.0 <br/>
numpy==1.14.5

## Running
python3 path/to/desired/model.py

## Future Plans
1. Saving checkpoints so that models can be trained and retrained
