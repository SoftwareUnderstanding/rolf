# WNUT-2020

### Contents

This directory sets out the model that formed our submission to the Workshop on Noisy User-generated Text 2020 Task 2: Identification of Informative COVID-19 English Tweets. 

We include:

* The model - WnutPytorchTransformers.py
* The script used to schedule running experiments

The data used in respect of this work is obtainable from https://github.com/VinAIResearch/COVID19Tweet

### Information

Using this model we acheived the following F1 Score against the holdout test set. 

| Rank | Team               | F1 Score |
| ---- | ------------------ | -------- |
| 1.   | NutCracker         | 0.9096   |
| 2.   | NLP\_North         | 0.9096   |
| ...  | ...                | ...      |
| 21.  | cxp949 (this work) | 0.8910   |
| ...  | ...                | ...      |
| 55.  | TMU-COVID19        | 0.5000   |

In achieving this result we ran WnutPytorchTransformers.py with the following arguments on an AWS EC2 p2 spot instance:

`RoBERTa roberta-base 4 128 4 1e-5 32 0.2 0.3 True True 2 6000 False False`

This set of hyperparameters were selected due to receiving the highest F1 score performance against the validation dataset. The ranges of hyperparameter values tested against the validation data to find the best performing model are presented below. 

### Hyperparameter Optimization

The hyperparameter values used to test the model against the validation data were informed by the hyperparameters for finetuning RoBERTa on the RACE, SQuAD and GLUE datasets (https://arxiv.org/abs/1907.11692). 

| Hyperparameter      | Values              |
| ------------------- | ------------------- |
| Epochs              | (2, 3, 4)           |
| Learning Rate       | (1e- 5, 2e-5, 3e-5) |
| Batch Size          | 32                  |
| Dropout Probability | (0.1, 0.2)          |
| Max TFIDF Features  | (6000, 9000)        |

