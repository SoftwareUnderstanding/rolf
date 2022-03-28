# Introduction
**BERT** (Bidirectional Encoder Representations from Transformers) is a NLP model developed by Google that follows the structure of a transformer. That structure was used to create models that NLP practicioners can then download and use for free such as RoBerta, XLNet or AlBert. You can either use these models to extract high quality language features from your text data, or you can fine-tune these models on a specific task (classification, entity recognition, question answering, etc.) with your own data to produce state of the art predictions.

In this repository, we will try to modify and fine-tune BERT to create a powerful NLP model for Question Answering, which means giving a text and a question, the model will be able to find the answer to the question in the text.

The dataset that we will be using is **SQuAD 2.0**. The dataset consist in a series of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage (sometimes the question might be unanswerable).

# Using  the tranformers library by HuggingFace

We first attempted to run the scripts provided by the transformers library by HuggingFace that fine-tune the base model of BERT in order to train it for Question Answering. We have used a small portion of the SQuAD dataset because our environment's performance didn't allow us to run the training on the whole dataset. (Both the colab and local environment we tried unfortunately ended by crashing)

The script that fine-tunes BERT using the library transformer and run_squad is: Nlp-squad-runner.ipynb

# Implementing the fine-tuning of BERT

After getting an insight into the challenges that we have to face, we've tried to implement the training and evaluation loops to fine-tune BERT with the SQuAD 2.0 dataset. We've analyzed the script given by the library transformers (run_squad.py) and we've attempted to re-create a fine-tuning more adapted to our needs. These are the steps that we've followed to fine-tune BERT.

1. Download the SQuAD 2.0 dataset (json files)
2. Transform our dataset into the format that BERT can be trained on by:
    * Apply the BERT tokenizer to our data.
    * Add special tokens to the start and end of each sentence.
    * Padd & truncate all sentences to a single constant length.
    * Map the tokens to their IDs. (the last 3 steps are done by a transformers function provided in order to extract features)
3. Load the base model of BERT. In our case, we chosed to use the BertForQuestionAnswering structures with some bert pretrained weights on the classic Bert part of it.
4. Train the BERT model with our SQuAD dataset (A reduced version in a first place).
5. Evaluate the model with the dev file provided by the website of Squad V2.
    
By using 10 000 text-question-answer, the precision of our model was: **49%**
![console screenshot](img/eval.PNG)

A save of this model can be found in the directory finetuned_squad_saved_acc_49 with our prediction files too

![console screenshot](img/loss.png)

The script that fine-tunes BERT using the library transformer is: Bert_fine_tune.ipynb

We have tried to launch the fine tuning with all the data but it was too long (16-17 hours) and our computer was literaly burning after an hour of it (the train interrupted after an hour resulted in an accuracy of 56%). As our script seem to train well, we think that a full training should give satisfying performances.

# Resources

- HuggingFace/transformers: https://github.com/huggingface/transformers 
- SQuAD 2.0: https://rajpurkar.github.io/SQuAD-explorer/

- Painless Fine-Tuning of BERT in Pytorch: https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
- BERT Fine-Tuning Tutorial with PyTorch: http://mccormickml.com/2019/07/22/BERT-fine-tuning/
- BERT for Question Answering on SQuAD 2.0: https://web.stanford.edu/class/cs224n/reports/default/15848021.pdf
