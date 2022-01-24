## General Description

This is an attempt to implement in Keras the Dynamic Memory Network described in
https://arxiv.org/pdf/1506.07285.pdf

Currently it achieves zero test error on tasks 1,4,6. I have not tested it on the bigger tasks,
due to computational constraints

## What do I need

- python 3.6+
- Keras 2.0+
- Tensorflow 1.2+
- Numpy


- A file that holds pre-trained embeddings, such as can be found on https://nlp.stanford.edu/projects/glove/
- A file that contains a training task in the format used by babi. You know just use babi - https://research.fb.com/downloads/babi/
- (Optional) If you want to be proper, a separate test task in the format used by babi - https://research.fb.com/downloads/babi/

## How do I run it?

run `python train_and_eval.py --settings <path_to_settings>` Where `<path_to_settings>`
points to a json file that follows the sample_experiment_settings.json pattern.
Make sure to select batch_size so that your trainset can be neatly divided into equal sized batches.
Alternatively just modify train_and_eval.py - Im not your mom, you can do whatever you want.

The training process will create a log where it will track training progress and
save the weights of the best models. If in the settings you supply a path to a test
task, in the end of training the model will be evaluated on that and output the
test accuracy and test loss

## Implementation notes

Read the paper at https://arxiv.org/pdf/1506.07285.pdf and https://arxiv.org/pdf/1603.01417.pdf

The main difference with the paper, is that the output and labels here are in the
form of one-hot vectors. That is the network is trained to output the correct answer out of a list of answers,
rather than generate the answer on its own (i.e. output a word vector).

I used this implementation: https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow as inspiration for the Episodic memory module,
so please be sure to check it out.

Preprocessing is a modified version of the examples given with Keras.  

## How far along is everything?

- You should be able to run and train on the different babi tasks.

- Training fails if the last batch is smaller. Working on it. Until then make sure
  your batch size neatly divides the train/test set.

- The code still needs some polishing and some documentation, and might be clunky.
So that part is still work in progress.

- Checks and verifications are kind of missing, so you can break things easily. Why would you want to do that?

- Missing dedicated option for evaluating an existing model

- You might run into problems with loading the saved weights - I am still working on that.

## It does not work!/Your code sucks!/Teach me your ways!/Why dont you try adding X?

Open an issue, send me a message, etc.
