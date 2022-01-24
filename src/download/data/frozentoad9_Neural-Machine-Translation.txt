# Neural-Machine-Translation

***This project was done as a part of a hackathon conducted by IIT Bombay***

This is an implementation of Neural Machine Translation using Encoder-Decoder Mechanism along with Attention Mechanism -  (https://arxiv.org/pdf/1409.0473.pdf) introduced in 2016.
The Encoder-decoder architecture in general uses an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation. This paper conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly.

### Encoder :
The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.


### Decoder :
In the simplest seq2seq decoder we use only last output of the encoder. This last output is sometimes called the context vector as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.
At every step of decoding, the decoder is given an input token and hidden state. The initial input token is the start-of-string <SOS> token, and the first hidden state is the context vector (the encoder’s last hidden state).

### Attention Decoder:
Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs. First we calculate a set of attention weights. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called attn_applied in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.
Calculating the attention weights is done with another feed-forward layer, using the decoder’s input and hidden state as inputs. Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

#### Usage

This will take the file kept in **data** folder. Default eng-hin.txt. 

    run : python main.py

In **main.py** enter your input sentence in source language(lang1) - line 55 to see the translation.
