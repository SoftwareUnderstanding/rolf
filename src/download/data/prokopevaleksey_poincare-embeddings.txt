# Poincaré Embeddings for Learning Hierarchical Representations

![demo.png](animation0.gif)


Here is my implementation of Poincare Embedding [https://arxiv.org/pdf/1705.08039.pdf]

The main idea of the paper is to use Hyperbolic space for embedding hierarchical data. 

Naturally, we want to represent data which have some sort of hierarchy in the way that related objects would be located in the neighbourhood of one another. 

As we go down the hierarchical tree the number of objects in that neighbourhoods increases exponentially.

If we want to fit these objects in Euclidean space we have to dramatically increase the dimensionality. 

Usage of Hyperbolic space solves this specific problem and authors claim to archive impressive results.

However, recreating the results came up to be a challenging task.

This implementation is based on original repo by Facebook Research https://github.com/facebookresearch/poincare-embeddings/

## Instalation
- requirements: Python 3.6
- clone the repo by
```
git clone https://github.com/prokopevaleksey/poincare-embeddings.git
```
- install dependencies by
```
pip install -r pip install -r requirements.txt  
```
## Run demo
 - to generate report run
 ```
 python vis.py 
 ```

 - to train the demo on mammal subset of WordNet and generate report run
 ```
 python train_demo.py
 ```
## Content
- data/wordnet --> contains .tsv files with data extracted from wordnet
  - mammal_closure.tsv --> WordNet synset closures used in original repo
  - mammal_hierarchy.tsv --> direct hyponyms of the synsets used in mammal_closure.tsv
  
- datasets.py --> PyTorch Dataset class for data parsing

- poincare.py --> Poincare distance function, mmbedding model and optimizer

- train_demo.py --> trains the model on toy data

- vis.py --> visualize results using plotly

- demo/ --> stores trained .pt models and .html reports

## TODO
- recreate results from the paper
- try different datasets
- explore metaparameters space
- organize and clean up
- improve visualization
- try to leveraze GPU
