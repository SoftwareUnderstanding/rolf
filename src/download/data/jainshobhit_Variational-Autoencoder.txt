# Variational-Autoencoder
This repository is an MxNet Gluon implementation of Variational Autoencoders(https://arxiv.org/abs/1312.6114) applied on a corpus of documents. 

The code was tested by running it on the NYTimes dataset dowloaded from https://archive.ics.uci.edu/ml/datasets/bag+of+words. The above mentioned link contains the NYTimes data in Bag of Words format. For faster loading of the data, the data can be converted to sparse matrix format and saved to a file using the .npz format. 
The following commands can be used for this purpose:
```
row, col, data = np.loadtxt('./docword_nytimes.txt', dtype = 'int32', skiprows = 3, unpack = True)
csr = csr_matrix( (data,(row,col)))
scipy.sparse.save_npz('nytimes_sparse.npz', csr)
```
The code can be tested by running it as it is. 
To run the code on cpu, change the context from mx.gpu() to mx.cpu() at [Line 110](https://github.com/jainshobhit/Variational-Autoencoder/blob/b6635471bc366d42f50b996c72d5f6e4686712a3/train.py#L110)

Variational Autoencoders can also be used as an **unsupervised learning technique to generate a latent representation of a document which represents its semantic contents**. If a bag of words representation of a document is given as an input to the model, the model learns a latent representation of the document. This latent representation can be considered as a probability distribution over a set of topics. Each topic can also be represented as a probability distrbution over words in a vocabulary, which is basically the decoder weights. For more mathematical details, please refer to https://arxiv.org/pdf/1511.06038.pdf.

## Requirements

The code is written in Python and uses [MXNet Gluon framework](https://mxnet.apache.org/). A GPU is not necessary and the code can be run on a CPU but running on GPU will give a significant boost to the speed. 

## Usage

To run the code, simply run 
```
python train.py
```
