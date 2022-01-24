# Research for Science seminar in HSE

<https://colab.research.google.com/drive/1ZadHBHjtZLh-9OqXYuH2QehIC8QUTkTg>

This is a notebook for experimenting for a pet project. The idea is to fine-tune an NLP model(e.x. AlBERT) to predict intention from user's requests for voice asistant like Alexa or Siri.

I'm going to use a pretrained model from tensorflow_hub as a backbone for this experiment, add several layers on top of it and fine-tune the model.

After that, the structured repository with all scripts for getting the data, preprocessing, setting up a model, training and inferencing will be prepared.

Each cell of this particular notebook can be treated as a separate python file for now(As I mentioned, I'll convert it into the full-featured project later.)

## Dataset and task

I would love to train a model that can predict person's intent from the message:

[Dataset](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines)

[More about the data in this blog post](https://medium.com/snips-ai/benchmarking-natural-language-understanding-systems-google-facebook-microsoft-and-snips-2b8ddcf9fb19)

[Archiv article](https://arxiv.org/abs/1805.10190)

## Structure of this notebook

- import required packages(Here I'll keep all of them at the top in the single cell)
- Download, store and preprocess data
- Set up the model
  - AlBERT is being used here <https://arxiv.org/pdf/1909.11942.pdf>
  - Download checkpoints for backbone(<https://github.com/tensorflow/models/tree/master/official/nlp/albert)>
  - Freeze it as I'm planning to use its embeddings as it is for now.
  - Set up the model's head
- Train the model
  - Prepare a pipeline to finetune the model from the previous step
  - I'm planning to use all features from tensorflow: Datasets operations, logging to tensorboard, checkpoints etc.
  - Freeze the graph for further inference
- Inference
  - Prepare a script to inference the final model
  - I would like to play a bit with optimization techniques from tensorflow package, so I'll add additional scripts for prunning and quantization using native tf tools.

===

## Some plots

### Confusion matrix

![](misc/conf_matrix-2.png)

### Accuracy by epoch

![](misc/epoch_acc.svg)

### Loss by epoch

![](misc/epoch_loss.svg)

===

## How to use this repo

### Install the requirements

'''
pip install -r requirements.txt
'''

### Download data

'''
sh data/download.sh
'''

### Train the model

'''
python tf_bert_sentiment/train.py
'''

### Infer on test data

'''
python tf_bert_sentiment/infer.py
'''
