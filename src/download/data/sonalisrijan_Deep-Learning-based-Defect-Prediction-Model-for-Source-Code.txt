# Deep Learning based Defect Prediction Model for Source Code

This project is a line-level defect prediction model for software source code from scratch. Line level defect classifiers predict which lines in a code are likely to be buggy. 

The data used for this project has been scraped from multiple GitHub repositories, and organized into dataframes with the following four columns:
1. instance: the line under test
2. context before: the context of the line under test right before the line. In this question, the context before consists of all the lines in the functions before the tested line.
3. context after: the context of the line under test right after the line. In this question, the context after consists of all the lines in the functions after the tested line.
4. is buggy: the label of the line tested. 0 means the line is not buggy, 1 means the line is buggy

We used Bidirectional Encoder Representations from Transformers, aka BERT in order to better capture the context of the tokens in the instance. 
BERT is a trained transformer encoder stack that applies bidirectional (rather non-directional) training in Transformer (a popular attention model), and has been immensely popular in encoding text (both software as well as natural language). BERT was published in 2018 by Devlin at al at Google AI (link: https://arxiv.org/pdf/1810.04805.pdf). BERT was inspired by the Transformer model that was first suggested in the paper: Attention Is All You Need (link: https://arxiv.org/pdf/1706.03762.pdf). 
The HuggingFace is an open-source community that has made available architectures from NLP in its Transformers library (includes BERT as well). We used HuggingFace transformers library to use BERT in our project while using the Pytorch framework (link: https://huggingface.co/transformers/model_doc/bert.html).

The BERT model we use has 4 attention heads, and 2 transformer layers (blocks). Along with the preprocessed tokens, we also made use of:
1. positional embeddings (0,1..1000): Conveys index of word/token
2. token types (0,1,2): Conveys type of token: 0 if token lies before <START>; 1 if token is between <START> and <END>; 2 if token is after <END>
3. attention mask (1/0) to convey: Conveys where padding has been done
