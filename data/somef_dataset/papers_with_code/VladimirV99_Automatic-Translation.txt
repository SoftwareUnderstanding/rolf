# Automatic-Translation

Automatic language translation using a sequence-to-sequence LSTM model

## Required system packages

- python
- pip
- graphviz

## Required libraries

- notebook
- numpy
- pandas
- tensorflow
- pydot
- nltk
- scikit-learn
- matplotlib

If you have conda installed, you can create an evironment with all required packages installed by running the following commands
```bash
conda env create -f environment.yml
conda activate translation
```

## Datasets

English word list: https://github.com/dwyl/english-words

French word list: http://www.lexique.org/

English-to-French translation datasets: 
- http://www.manythings.org/anki/
- https://www.tensorflow.org/datasets/catalog/wmt14_translate

## Bibliography

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- https://nlp.stanford.edu/projects/glove/
- https://google.github.io/seq2seq/
- https://keras.io/examples/nlp/lstm_seq2seq/
- https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
- https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/
- https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
- https://medium.com/@d.salvaggio/sequence-to-sequence-architectures-ad6ff4451f84
- https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639
- https://towardsdatascience.com/neural-machine-translation-using-seq2seq-with-keras-c23540453c74
