# README (TOPICS/TOPIC LIST)

## Notebook 1: An Introduction to NLP and Core Linguistic Concepts
  - Outline and breakdown of notebooks and how they're structured and what you can find in each
  - Introduction to Syntax, semantics, and pragmatics (definitions, examples of the differences using project gutenburg)
    - include references to English syntax and a brief overview of the key terminologies with regards to English syntax with examples using code to extract them
    - Outline POS, phrase structure etc.
  - Ambiguity (examples with project gutenburg text) and Compositionality (with code examples of compositionality, again using project gutenburg free texts)
  - Briefly explain how these cause complications for ML with regard to NL with demonstrable examples from the Gutenberg texts
  
#### Exercises:

    1. Summarize the types of ambiguity and give a brief explanation of each.

    2. Give an example of each type of ambiguity not included in this notebook from the text data loaded above.

    3. How does compositionality affect meaning and make natural language processing harder?

    4. Label all the parts of speech of the following sentence and identify the noun phrase, predicate, complement, and prepositional phrases along with their objects. ..
    "Suzy was determined to eat the lamb chop; her fork and knife were primed and gripped tightly in her hands, after which she dove in and did not look up once from her meal."

    5. Give examples of every part of speech with an excerpt from the text we loaded in this section.

#### Reference Papers/Links
    
    1. Ambiguities in Natural Language Processing (Anjali et al, 2014) | http://www.ijircce.com/upload/2014/sacaim/59_Paper%2027.pdf

    2. https://www.nlpworld.co.uk/nlp-glossary/a/ambiguity/

    3. https://en.wikipedia.org/wiki/Principle_of_compositionality
    
    4. https://plato.stanford.edu/entries/compositionality/

---
---

## Notebook 2: The Basics of Word Embeddings
  - Why do we use word embeddings?
  - The difference between Distributed and Distributional embeddings
  - An easy way of approaching word embeddings using count-based methods
  - Walkthrough of simple one-hot representations, building till we get to Word2Vec and GLoVe embeddings
  - simple walkthrough of training your own word embedding using a given corpora
    - followed by using GluonNLP for a simple OOTB approach for retrieving WE
  
#### Exercises:
  
    1. For your specific native language, if not English, how would word embeddings work? For symbolic languages (Chinese, Thai, Korean etc.)?
    
    2. What are possible avenues for improving these word embeddings?
    
    3. Is there any way we can reduce training time? Is it possible to paralellize this training of word embeddings?
    
    4. What is the basic purpose of word embeddings?
    
### Reference Papers
    
    1. Wordnet | https://wordnet.princeton.edu/

    2. Linguistic Regularities in Continuous Representations (Mikolov et al. 2013) | http://aclweb.org/anthology/N13-1090
    
---
---

## Notebook 3: Word Embedding Evaluation and Improvement
  - Evaluation metrics
    - intrinsic vs Extrinsic + qualitative vs quantitative
    - word similarity and analogies
  - Logical walkthrough of why the evaluation metrics invalidate/date some of the simpler word embedding concepts
  - SVD, LSA, PCA as a solution to compact the word embeddings and dealing with the sparsity problem
  - Fasttext embeddings and sub-word embeddings.
  - Issue with bias in word embeddings and how to "debias" Word Embeddings
  - Systematic improvements for faster training
    - Introduction to parallel training
    - Softmax approximations
 

#### Exercises:
  
    1. What are the differences and improvements of the word embeddings discussed in this notebook versus the ones discussed in the previous word embedding notebook?
    
    2. What are other evaluation metrics that we can use to understand the quality of word embeddings? How has that influenced the start of sub-word embeddings? How do those differ from normal word embeddings?

### Reference Papers
    
    1. Evaluating Word Embeddings (Schnabel et al. 2015) | http://aclweb.org/anthology/D15-1036

    2. Character-based Embeddings (Ling et al. 2015) | http://aclweb.org/anthology/D15-1176

    3. Subword-based Embeddings (Bojankowski et al. 2017) | http://aclweb.org/anthology/Q/Q17/Q17-1010.pdf
    
    4. Cross-lingual Embeddings (Faruqui et al. 2014) | http://aclweb.org/anthology/E/E14/E14-1049.pdf
    
    5. Sparse Word Embeddings (Murphy et al. 2012) | http://aclanthology.info/pdf/C/C12/C12-1118.pdf
    
    6. De-biasing Word Embeddings (Bolukbasi et al. 2016) | https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf
    
    7. Morphology-based Embeddings (Luong et al. 2013) | http://aclweb.org/anthology/W/W13/W13-3512.pdf

---
---

## Notebook 4: Data Preparation for NLP and Sentence-level Embeddings
  - Introduce the data preparation tools and simple implementations along with their respective complications
    - tokenization
    - batching
    - lemmatization/stemming
    - word and sentence segmentation
    - BPE
  - Talk about the task of sentiment analysis and the challenges that need to be resolved before accomplishing this task from the basic word embedding information gleaned from this section
  - Comparison of BOW + Word2Vec vs TF-IDF + Word2Vec vs CNN + Word2Vec vs RNN+Word2Vec for the task of sentiment analysis
  - complete walkthrough of using all of these different methods for sentiment analysis and understanding sentence level context and embeddings
  
  
#### Exercises:
  
    1. What are the basic tasks that need to be done for data preparation and how does this differ from other non-NLP ETL?
    
    2. What is a key difference between certain symbolic languages (Chinese, Japanese, Thai etc.) and English when it comes to data preparation and preprocessing?
    
    3. How can we improve this basic sentiment analysis model?
    
    4. What factors have we not considered using this more naive word embedding implementation? How can we improve the word embedding part? 
    
    5. Redo sentiment analysis with one of the models (BOW/CNN/TF-IDF/RNNs) but use a more complex word embedding; how does this affect the results?

### Reference Papers
    
    1. Convolutional Neural Networks (LeCun et al. 1998) | http://www.cs.nyu.edu/~yann/2008f-G22-2565-001/diglib/lecun-98.pdf

    2. CNNs for Text (Collobert and Weston 2011) | https://arxiv.org/pdf/1103.0398.pdf

    3. CNN for Modeling Sentences (Kalchbrenner et al. 2014) | http://aclweb.org/anthology/P/P14/P14-1062.pdf
    
    4. Learning Syntax from Translation (Shi et al. 2016) | http://aclweb.org/anthology/D/D16/D16-1159.pdf
    
    5. Learning Sentiment from LMs (Radford et al. 2017) | https://arxiv.org/pdf/1704.01444.pdf
    
    6. Semantic Relatedness (Marelli et al. 2014) | http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf
    
    7. Techniques and Applications for Sentiment Analysis (Feldman et. al 2013) | https://cacm.acm.org/magazines/2013/4/162501-techniques-and-applications-for-sentiment-analysis/abstract

---
---

## Notebook 5: Language Models
  - Construction of simple N-gram (Unigram + bigram) models
    - Introduce the concept of evaluation and metrics (Likelihood and Perplexity)
  - Simple feed-forward NN Language model building
  - Demonstrate complications, and the solutions:
    - Smoothing
    - Backoff
  - Char/subword/Word level embeddings, and the effect on the language model

#### Exercises:
    
    1. Construct an n-gram model with all of the bells and whistles of smoothing and backoff but using character level embeddings. How much better or worse does this perform compared to the simple models? Using sub-word embeddings, do you do better or worse?

    2. How can we improve these language models? Are there any simple improvements that will change the outcome of the models?
    
### Reference Papers
    
    1. Maximum entropy (log-linear) language models. (Rosenfeld 1996) | http://repository.cmu.edu/cgi/viewcontent.cgi?article=2337&context=compsci

    2. A Neural Probabilistic Language Model. (Bengio et al. 2003, JMLR) |  http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

    3. Using the Output Embedding. (Press and Wolf 2016) | https://arxiv.org/pdf/1608.05859.pdf
    
    4. Discourse Language Models (Chaturvedi et al. 2017) | http://www.aclweb.org/anthology/D17-1168

---
---
    
## **Extras**:
Glossary of NLP terms and NLP tasks with brief explanations of each, easily searchable

### **Chapter Exercises**
- What are word embeddings and how do they work?
- How can we simply represent an entire sentence using embeddings?
- What are the differences between syntax, semantics and pragmatics?
- What are subwords and character-based embeddings?
- What's the difference between a distributional and distributed representation?
- What's the difference between extrinsic and intrinsic evlauations?
- Model the steps for accomplishing machine translation, including steps for data preparation and the variety of subtasks needed to train a model.
- What do word similarity metrics and word analogy metrics for word embeddings show us?
- How do ambiguity and compositionality affect NLP?
- What are the different types of ambiguity?
- How can we use compositionality to our advantage?

