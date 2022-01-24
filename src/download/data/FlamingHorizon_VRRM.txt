Learning to Activate Logic Rules for Textual Reasoning    
============================================================================  
**Note that:**  
---------------------------------------------------------------------------- 
**Usage:**  
    1. Please download the software and put bAbI-20 dataset in ../en/    
    2. Use THEANO_FLAGS=devices=cpu/gpu python run-qa-*.py to run the main functions.      
    
**Notices:**  
    1. The suggested memory of machine is 16GB RAM;    
    2. The code depends on Theano and Keras framework.    
    3. In qa17 and 18, two or more words may represent only one recognized entity in a sentence. We concate them into one word and generate new files named ''qa*_concat_train/test.txt''.    
    4. The bAbI-20 dataset is described in the paper:(https://arxiv.org/abs/1410.3916)    
