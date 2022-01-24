# Word Vectors

A repository to explore dense representations of words.

## Introduction

- Most words are symbols for an extra-linguistic entity - a word is a signifier that maps to a signified (idea/thing)
- Approx. 13m words in English language
  - There is probably some N-dimensional space (such that N << 13m) that is sufficient to encode all semantics of our language
- Most simple word vector - one-hot encoding
  - Denotational semantics - the concept of representing an idea as a symbol - a word or one-hot vector - sparse, cannot capture similarity - localist encoding

Evaluation

- Intrinsic - evaluation on a specific, intermediate task
  - Fast to compute
  - Aids with understanding of the system
  - Needs to be correlated with real task to provide a good measure of usefulness
  - Word analogies - popular intrinsic evaluation method for word vectors
    - Semantic - e.g. King/Man | Queen/Woman
    - Syntactic - e.g. big/biggest | fast/fastest
- Extrinsic - evaluation on a real task
  - Slow
  - May not be clear whether the problem with low performance is related to a particular subsystem, other subsystems, or interactions between subsystems
  - If a subsystem is replaced and performance improves, the change is likely to be good

## Useful Links

- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)
- [http://mccormickml.com/](http://mccormickml.com/)
- [https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#examples-word2vec-on-game-of-thrones](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#examples-word2vec-on-game-of-thrones)

## Models

- [SVD-based](notebooks/svd_word_vectors.ipynb)
- LSA
- LDA
- word2vec
- GloVe

## SVD-based methods

1. Loop over dataset and accumulate word co-occurrence counts in a matrix $X$
2. Perform a SVD on $X$ to get a $USV^T$ decomposition
3. Use the rows of $U$ as the word embeddings (use the first $k$ columns to limit the embedding dimension)

Methods to compute $X$:

- Word-Document matrix
  - Each time word $i$ appears in document $j$, increment $X_{ij}$
  - $X \in \mathcal{R}^{V \times M}$, where $M$ is the number of documents
- Window-based co-occurrence
  - $X \in \mathcal{R}^{V \times V}$

Variance captured by embeddings:

$$
\frac{\sum_{i=1}^k\sigma_i}{\sum_{i=1}^{V \textnormal{ or } M}\sigma_i}
$$

Problems:

- Vocab fixed at start - based on corpus
- Matrix is sparse
- Matrix is high-dimensional (quadratic cost for SVD)
- Need to perform some hacks to adjust for imbalanced word frequencies

Solutions to issues:

- Ignore function words
- Weight co-occurrence counts based on distance between words in the document
- Use Pearson correlation and set negative counts to 0 instead of using just the raw count

## word2vec

### References

1. [2013, Mikolov et al. Efficient Estimation of Word Representations in Vector Space. arxiv:1307.3781v3](https://arxiv.org/pdf/1301.3781.pdf)
2. [2013, Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality. NIPS 2013.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### Theory

Two model architectures:

1. Continuous Bag-of-Words (CBOW) - uses context words to predict target word
2. Continuous Skip-gram - uses target word to predict context word

Training methods

- Negative sampling - include negative examples in computation of cost function
  - Better for frequent words and lower dimensional vectors
- Hierarchical softmax - define objective using an efficient tree structure to compute probabilities for the complete vocabulary
  - Better for infrequent words

Objective function (cross-entropy)

$$
H(\hat{y}, y) = -\sum_{w\in W}y_j\log(\hat{y_j})
$$

Since $y_j = 1$ for the target word and $y_j = 0$ for all other words, the equation reduces to

$$
H = -\log(\hat{y})
$$

where $\hat{y}$ is the predicted probability of the target word.

### Skip-gram Model

#### Objective function

Assumption: Given the center word, all output words are completely independent (Naive Bayes assumption)

$$
\textnormal{Likelihood} = L(\theta) = \prod_{t=1}^{T}\prod_{-c\leq j\leq c, j\neq 0}P(w_{t+j}|w_t;\theta)
$$

Use negative log-likelihood for better scaling

$$
J(\theta) = -\frac{1}{T}\log L(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\leq j\leq c, j\neq 0}\log P(w_{t+j}|w_t;\theta)
$$

- $c$ is the size of the training context (which can be a function of the center word $w$)
- Larger $c$ - more training examples, higher accuracy, increased training time

The probability $P(w_{t+j}|w_t)$ is calculated using the softmax function:

$$
P(w_O|w_I) = \frac{\exp(u_O^Tv_I)}{\sum_{w=1}^{W}\exp(u_w^Tv_I)}
$$

- $u_w$ and $v_w$ are the 'input' and 'output' vector representations of $w$, and $W$ is the size of the vocabulary
- This formulation is impractical computationally because it requires computing the softmax over all the representations in the vocabulary

#### Gradient

$$
\begin{aligned}
    \frac{\partial P(w_O|w_I)}{\partial v_I} &= u_O - \sum_{w\in V}P(w_w|w_I)\cdot u_w \\
    \frac{\partial P(w_O|w_I)}{\partial u_O} &= v_I - v_I\cdot P(w_O|w_I) \\
    \frac{\partial P(w_O|w_I)}{\partial u_{w, w\in V, w\neq O}} &= -v_I\cdot P(w_w|w_I)
\end{aligned}
$$

### CBOW Model

The probability $P(w_t|w_c)$ is calculated using the softmax function:

$$
\textnormal{Likelihood} = L(\theta) = \prod_{t=1}^{T}P(w_t|\{w_j\}_{-c\leq j\leq c, j\neq 0};\theta)
$$

$$
P(w_I|w_C) = \frac{\exp(u_I^Tv_C)}{\sum_{w=1}^{W}\exp(u_w^Tv_C)}
$$

where $v_C$ is the sum of 'output' representations of all words in the context window:

$$
v_C = \sum_{-c\leq j\leq c, j\neq 0}v_j
$$

#### Objective function

$$
J(\theta) = -\frac{1}{T}\sum_{t=1}^T\log P(w_I|w_C)
$$

#### Gradient

$$
\begin{aligned}
    \frac{\partial P(w_I|w_C)}{\partial v_{c,j}} &= u_I - \sum_{w\in V}P(w_I|w_C)\cdot u_w \\
    \frac{\partial P(w_I|w_C)}{\partial u_I} &= v_C - v_C\cdot P(w_I|w_C) \\
    \frac{\partial P(w_I|w_C)}{\partial u_{w, w\in V, w\neq I}} &= -v_C\cdot P(w_I|w_C)
\end{aligned}
$$

## GloVe

- Combines results from LSA with word2vec
- Predict probability of word $j$ occuring in the context of word $i$ using global statistics (e.g. co-occurrence counts)