# RegNet-TF

This is an unofficial implementation of RegNet [(Designing Network Design Spaces)](https://arxiv.org/abs/2003.13678).    

## Requirements
(TODO : requirements.txt and Dockerfile for the image of fixed environment.)
- python >= 3.6
- tensorflow >= 2.2

## Training
### Set design space
- First, set the design space in `config.yml`
- Second, sample models in the design space
```
>>> python search_space.py --config /path/of/config.yml --num_model $NUM_MODEL --model_name $MODEL_NAME
```
or
```
>>> ./search.sh
```
- Then, train models which are already set
```
>>> ./main.sh
```

## Evaluation
(TODO : shell command for evluation)
