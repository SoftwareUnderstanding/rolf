
* REPO IN CONSTRUCTION

This repo implements a language model based on the word level. 

This code is meant to work with heavy datasets (>500 Go of text), so we optimized the data loading and processing. See `Lazy contiguous dataset`

Through the code you will see some checks 
```
if os.name=='nt': 
  ...
 ```
 Which is a statement that allows to check if the code is run on a local windows machine for debugging or on a Linux cluster.
 
 The small dataset used to debug/optimize this project can be found at : https://www.kaggle.com/c/asap-sas/data
 
 
 

* Large vocabulary

Also, as the vocabulary size is large (~100K to 1M words similar to the 1B words benchmark) we implemented an efficient softmax + Cross Entropy based on : https://arxiv.org/abs/1609.04309


