## Identifying Similar Test Cases That Are Specified in Natural Language

This repository contains the source code of our technique and related experiments to identify similar test cases written in natural language. The technique first clusters test steps which are semantically similar and then uses those clusters to identify similar test cases.

</br>

To cluster similar test steps, we performed several experiments with the following text embedding techniques, text similarity metrics, and clustering algorithms:

**Text embedding techniques**

* [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)
* [BERT](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)
* [Sentence-BERT](https://arxiv.org/pdf/1908.10084.pdf)
* [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)
* TF-IDF


**Text similarity metrics**

* [Word  Mover’s  Distance  (WMD) ](http://proceedings.mlr.press/v37/kusnerb15.pdf)
* Cosine score

**Clustering algorithms**

* Hierarchical Agglomerative Clustering
* K-means

</br>
To find similar test cases, we used the identified clusters of similar test steps to build and evaluate four different techniques.


---


## Structure of directories
 
 The following directories contains the source code of all the approaches that were part of our experiments. 


 - [test-step-clustering](/test-step-clustering/): contains the notebooks with the source code for our test step clustering experiments.
 
 - [test-case-similarity](/test-case-similarity/): contains the notebooks with the source code for our test case similarity experiments.
 
 - [evaluations](/evaluations/): contains the notebooks with the source code to evaluate all the approaches for test step clustering and techniques for test case similarity.


---


## Dependencies

The following dependencies are required to run the notebooks on your local machine:

 - Python 3.7
 
  
 - [Numpy 1.19](https://numpy.org/)

    `
    pip install numpy
    `


 - [Pandas 1.1.5](https://pandas.pydata.org/)
 
    `
    pip install pandas
    `

  
 - [matplotlib 3.0.3](https://matplotlib.org/)

    `
    pip install matplotlib
    `
    
 
 - [scikit-learn 0.21.1](https://scikit-learn.org/stable/)

    `
    pip install scikit-learn
    `
    
 - [Gensim 3.8.3](https://radimrehurek.com/gensim/)

    `
    pip install gensim
    `

 - [NLTK 3.4.1](https://www.nltk.org/)

    `
    pip install nltk
    `
    
 - [Torch 1.7.1+cpu](https://pytorch.org/)

    `
    pip install torch
    `    
    
 - [Transformers 4.3.2](https://huggingface.co/transformers/)

    `
    pip install transformers
    `   
    
 - [Sentence Transformers 0.4.1](https://www.sbert.net/)

    `
    pip install sentence-transformers
    `  
    
 - [TensorFlow 2.4.1](https://www.tensorflow.org/)

    `
    pip install tensorflow
    `  
    
 - [TensorFlow_Hub 0.11.0](https://www.tensorflow.org/hub)

    `
    pip install tensorflow-hub
    `  
    
