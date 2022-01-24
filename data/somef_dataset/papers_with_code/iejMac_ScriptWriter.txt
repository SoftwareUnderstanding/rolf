# ScriptWriter

Neural network that fills in blank spaces in movie scripts

## Data:

## Pre-processing:

1. txt format -> clean txt format (removing useless tabs, repeated \n etc. as little format noise as possible) !!!
2. Dividing scripts into tokenized torch arrays randomly (needs to be more way efficient) !!!

## Models:

1. Contrastive
   input: context, fragment
   output: similarity score

2. Generative
   input: masked context (mask where fragment should start)
   output: context + fragment or context + new word + shifted mask

Notes:

-   Can't really use fully pretrained transformers since diff vocab because of formatting
-   Use some pretrained layers from good transformers

## Idea:

Contrastive model = value net

Generative model = policy net

Combine them with MCTS and generate many path's that the fragment could take

## TODO:

1. DATA TASKS:

-   Data extraction pipeline improve efficiency !
-   Data cleaning !!

2. TOKENIZATION !!!

-   How to tokenize data
-   How to include whitespace
-   In general figure out tokenization step
-   Do we need new vocab or can we just use one

## Useful papers for this project:

1. CLIP: https://arxiv.org/pdf/2103.00020.pdf

2. Transformers: https://arxiv.org/pdf/1706.03762.pdf

3. LayerNorm: https://arxiv.org/pdf/1607.06450.pdf
