# Paraphrase Identification in Russian

## Getting Started 

Application of NLP methods on russian dataset ParaPhraser for paraphrase identification problem.

## Methods

* Word2Vec + TF-IDF weighting
* Bilaterial Multi-perspective Matching (BiMPM)

## Guide

* /Data/ - includes all necessary data for solving task on ParaPhraser + scripts that make files with word embeddings (wordvec_rv_3.txt) and train/test data (train_PP.tsv and test_PP.tsv)
* /Word2Vec/ - includes scripts for method Word2Vec + TF-IDF weighting
* /BIMPM/ - includes scripts for method BiMPM

### Prerequisites

* Python 3.6
* Tensorflow 1.5

### Papers
[1] T. Mikolov "Efficient Estimation of Word Representations in Vector Space" - https://arxiv.org/pdf/1301.3781.pdf

[2] Zhiguo Wang, Wael Hamza, Radu Florian "Bilateral Multi-Perspective Matching for Natural Language Sentences" - https://arxiv.org/pdf/1702.03814.pdf
