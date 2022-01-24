# Korean-POS-Tagger-LSTM
Korean POS Tagger Using Character-Level Sequence to Sequence Model 

[1] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks." Advances in NIPS (2014), https://arxiv.org/abs/1409.3215.

[2] K. Cho, and et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.“, https://arxiv.org/pdf/1406.1078.pdf

[3] https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

Model is same as [3] but diffrent application. 

Character-Level Sequence to Sequence Model:

  Input sequences
  
    Korean text
    
    Corresponding Korean POS-tag text
    
  An encoder LSTM turns input sequences to 2 state vectors
  
    Preserve the last LSTM state and discard the outputs   
    
  A decoder LSTM is trained to the target POS-tag into the same sequence 
  
    Keep offset by one time-step for future
    
    The offset uses as initial state the state vectors from encoder
    
    The decoder learns to generate POS-tag [t+1] by a given POS-tag [t]
    
    
Inference from trained model:

  Encoder input sequences
  
    Encode the Korean text sequence into state vectors
    
    Start with a target sequence of size 1 (sequence of characters)
    
  Feed the state vectors and each character target sequence to the decoder
  
    To produce predictions for the next character
    
  Sample the next character using these predictions
  
    Apply argmax
    
    Append the sampled character to the target sequence
    
  Repeating until to reach the end-of-sequence character
  

Prerequisite:
==============
keras

numpy

pickle


For train script:
=================
python train.py --train_file train.txt

For test script:
================
python test.py --input_file test.txt --output_file result.txt

Files:
LSTM_KR_PoS.py :  Character embedding model for Korean part of speech tagging

train.py : Training script 

test.py: Testing script


For testing with already train model put all of the saved model files in a 
same directory as test.py 

Model File List:

decoder_model_pos_kr_v10000.json

decoder_model_weights_pos_kr_v10000.h5

encoder_input_data_v.data

encoder_model_pos_kr_v10000.json

encoder_model_weights_pos_kr_v10000.h5

input_texts_v.data

input_token_index_v.data

max_decoder_seq_length_v.data

num_decoder_tokens_v.data

s2s_pos_kr_v10000.h5

target_token_index_v.data


Train with valid.txt:

(Due to memory resource constraint in my computer, current saved model trained with a first 10000 data point from valid.txt files
)

Train with train.txt:

"Train_Save_Model_10000" folder contain all learned model using train.txt with a first 10000 data point. 

Model files Names:

decoder_model_pos_kr_10000.json

decoder_model_weights_pos_kr_10000.h5

encoder_input_data.data

encoder_model_pos_kr_10000.json

encoder_model_weights_pos_kr_10000.h5

input_texts.data

input_token_index.data

max_decoder_seq_length.data

num_decoder_tokens.data

s2s_pos_kr_10000.h5

target_token_index.data





