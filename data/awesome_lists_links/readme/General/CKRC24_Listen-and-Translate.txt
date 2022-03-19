# NTU Machine Learing 2017 Fall Project: Listen and Translate

This is a project of Machine Learning at NTU. Given a Taiwanese audio signal, select the most possible Chinese translations from the given options. We implemented  **Seq2Seq model** and **Retrieval model** on this task and ranked No.2 on the Kaggel competition website.

## Getting Started

The following instructions will get you a copy of the project and running on your local machine for testing purposes.

### Prerequisite & Toolkits

The following are some toolkits and their version you need to install for running this project

* [Python 3.6](https://www.python.org/downloads/release/python-360/) - The Python version used
* [Keras 2.1.1](https://pypi.python.org/pypi/Keras/2.1.1) - Deep Learning for Python
* [Tensorflow](https://www.tensorflow.org/install/) - Tensorflow backend for keras
* [Gensim 3.1.0](https://pypi.python.org/pypi/gensim/3.1.0) - Python framework for fast Vector Space Modelling

In addition, it is required to use **GPU** to run this project.

## Running the tests

The following are some instructions to reproduce the results of the model.

### Installation

To run the project, first clone the project and go into the folder

```
git clone ...
cd Listen-and-Translate/
```
### Training/Testing Dataset

We used the dataset on Kaggle as our training/testing dataset, and we did some preprocessing on these data, the preprocessed data are stored in the **data** folder.

### Model

We only provide the pre-trained **Retrieval Model** for reproducing the results since we discovered that retrieval model performs better accuracy on this tasks. The model is stored in the **model** folder.

### Source Codes & Report

To see our model structures, you can find the code in the **src** folder and further explainations in our **Report.pdf**.

### Reproducing results

To reproduce the results, please provide the **test.csv** file path as argument and run the **final.sh** script file. The reproduced csv file will output into the **prediction** folder.

```
bash <test.csv file path> final.sh
```

## Kaggle Competition Website

Our model ranked No.2 on public set and No.3 on private set.
This is the Kaggle [Competition Website](https://www.kaggle.com/c/ml2017fallfinaltaiwanese/leaderboard).

## Reference

[1] The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems Ryan Lowe el at.(2016)
https://arxiv.org/pdf/1506.08909.pdf <br />
[2] Attention-Based Models for Speech Recognition Jan Chorowski el at.(2015)
https://arxiv.org/abs/1506.07503
