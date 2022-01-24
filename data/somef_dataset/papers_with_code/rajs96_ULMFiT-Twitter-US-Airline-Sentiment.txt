# Demystifying ULMFiT - A state-of-the-art transfer learning approach for NLP

## The problem
Many people are interested in getting into machine learning research, but in practice, it can be difficult.  There are several reasons for this:
- Reading research papers is hard.
- It can be hard to find well-explained code implementations. Very high barrier to entry.
- As more complex methods are used, interpretability flies out the window. We start to forget why we're doing things, which is a very bad thing.
- Reading papers usually assumes background knowledge, which leads to the reader to another dilemma of "what do I need to understand to understand this concept, and what resources to I need to visit to gain this understanding?"

## The solution
The truth is, there isn't one solution. You'll learn a lot from grinding through paper implementations yourself. But there are certain strategies that can make the process easier and more fun: that's what I've tried to illustrate in this repository. In particular, I'm going to walk through how I would go about interpreting a ML research method and how I'd apply it to a new dataset.

More specifically, we're going to apply a ULMFiT approach to predict sentiment towards an airline based on a Tweet. The ULMFiT approach was devised by Jeremy Howard and Sebastian Ruder, and it's essentially a transfer learning approach for NLP.

**This repo focuses on the problem-solving, intuitions, and actual process of applying ULMFiT to a novel problem**.

**Important note:** This repo was not created to undermine or replace the work that Howard and Ruder have done. It's simply my own interpretation/explanation of their method, which I hope will help others gain a better intuition of cutting-edge machine learning research.

There are three notebooks in this repo that are of interest:

## Twitter US Airline Sentiment Classification - Understanding the problem background
(Background_info.ipynb)
In this notebook, we'll talk about domain of NLP, more specifically text classification. We'll understand what the initial approaches were, and why a ULMFiT approach is better. Reading this is key to understanding why language modeling is an effective transfer learning approach for NLP.

## ULMFiT Approach
(ulmfit_runthrough-explanation.ipynb)
A run-through of the ULMFiT approach to the Twitter dataset, and explanations throughout.

## ULMFiT Results/Discussion
(ulmfit_results.ipynb)
Functions to produce results with the Twitter dataset, and an analysis of our results (we don't go through every possible combination of hyperparameters, so most of it is just result reporting). We'll also talk about some of the shortcomings of the model and how we might improve going forward.

## References
- "Universal Language Model Fine-tuning for Text Classification": https://arxiv.org/abs/1801.06146
- "Regularizing and Optimizing LSTM Language Models": https://arxiv.org/abs/1708.02182
- "A Comparison of Pre-processing Techniques for Twitter Sentiment Analysis": https://link.springer.com/chapter/10.1007/978-3-319-67008-9_31
- "How transferable are features in deep neural networks?": https://arxiv.org/abs/1411.1792
- "A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay": https://arxiv.org/abs/1803.09820
- "Grokking Deep Learning" - https://www.manning.com/books/grokking-deep-learning
- Andrew Ng's Deep Learning course: https://www.coursera.org/specializations/deep-learning
- Stanford's Deep Learning for NLP course: https://cs224d.stanford.edu/
- fastai documentation: https://docs.fast.ai/
- fastai course: https://course.fast.ai/
- Finding an optimal learning rate: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html


## Some key corrections
### ulmfit_runthrough_explanation
- When fine-tuning the language model, when we initialize the Learner with 'language_model_learner', the comment above should read "pass in **drop_mult=0.3**" to specify that our dropouts for the model are with p = 0.3.
- In the code cell under "Getting back to the big picture, the comment should read "as discussed before, we choose **'1e-2'** because it's slightly **smaller** than the minimum loss LR"
