# Machine Learning to Predict Subcellular Location of Protein by their Amino Acid Sequence

In this project, I have used AWD-LSTMs and ULMFiT to train a neural network to classify proteins based on their amino acid sequence.

## Building my own RNN and LSTM network using Numpy

First, to understand the RNN and LSTM network I have build them from the ground using only Numpy.
The files can be found in the additional folder.

## Subcellular protein location

After making the RNN and LSTM myself. I used PyTorch to create the AWD-LSTM and after that the other parts needed for the complete classifier.

## Work in progress

* Train the Language model longer
* Use a reference paper for measuring if it improves a benchmark
* Improve the models
  * Adding dropouts
  * Last hidden layer
  * Improve pooling
  * Training in batches
  * Probably more to come

## References

><https://arxiv.org/abs/1708.02182> Regularizing and Optimizing LSTM Language Models
><https://arxiv.org/abs/1801.06146> Universal Language Model Fine-tuning for Text Classification
