# AML-CoVe
University of Oxford - Advanced Machine Learning Reproducibility Challenge. Contextualized Word Vectors (CoVe) - https://arxiv.org/pdf/1708.00107.pdf

Tensorflow 2.0-alpha and Keras 2.2.4 is used.


To train the models, first train the NMT. We used the WMT ('16) data to train, which can be found at http://www.statmt.org/wmt16/. 
Next, the CoVe encodings can be generated using the Get_Encodings script. 
Lastly, the weights are loaded into the BCN or CNN classification network. We use the SST data set, which can be found at https://nlp.stanford.edu/sentiment/ 

All models require pretrained GloVe embeddings, which we download from https://nlp.stanford.edu/projects/glove/. 

The LSTM-Encoder and BCN sentiment model follow McCann et al., see https://arxiv.org/pdf/1708.00107.pdf. 
The single layer and three layer CNN sentiment model follow Kim et. al (2014) and Ouyang et al. (2015), see https://ieeexplore.ieee.org/abstract/document/7363395 and https://arxiv.org/abs/1408.5882.
The transformer encoder is based on Vaswani et al. (2017), see https://arxiv.org/abs/1706.03762. 
