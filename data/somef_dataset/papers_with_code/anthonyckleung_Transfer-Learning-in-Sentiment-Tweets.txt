# Transfer-Learning-in-Sentiment-Tweets

**Task**: Demonstrate the effective and efficiency of transfer learning via the 
Universal Language Model Fine-tuning model in NLP applications

For this demonstration, the Twitter data on people's belief on global warming/climate is used. 
Note that the sentiment of the tweets are based on the contributors evaluation of the tweets.
This demonstration is done on a jupyter notebook.

### Prerequisites:
* Python 3
* Jupyter notebook
* fastai
* pytorch
* re
* numpy, seaborn, pandas
* data source: [Figure-Eight](https://www.figure-eight.com/data-for-everyone/)



### Description
This notebook focuses on the so-called Universal Language Model Fine-tuning (ULMFiT) introduced by 
Jeremy Howard and Sebastian Ruder [[1](https://arxiv.org/abs/1801.06146)]. 
The ULMFiT model consists of three main stages in the building a language model (LM):

1. **General-domain LM  pre-training**: Similar to the ImageNet database used in computer vision, 
the idea is to pre-train a large corpus of text. The ULMFiT has a pre-trained model called the 
`Wikitext-103` where more than 20,000 Wikipedia articles was trained on. 
This is alreadly included in the `fastai.text` API, thus, it is not necessary to carry out this step.

2. **LM fine-tuning**: Because the target data (typically) comes from a different 
distribution from the general-domain, it is necessary to fine-tune the LM to adapt to 
the idosyncrasies of the target data. Howard and Ruder suggested *discriminative fine-tuning*
and *slanted triangular learning rates* for fine-tuning the LM. These techniques are available 
in `fastai` and are described in [[1](https://arxiv.org/abs/1801.06146)].

3. **Classifier fine-tuning**: Using the updated weights from the previous step, a classifier
can be fine-tuned. Howard and Ruder suggested a few techniques which include *concat pooling* 
and *gradual unfreezing*. The latter in particular is used in this demonstration. 
Again, the `fastai` framework allows one to perform this technique.

**Reference**

[1] J. Howard and S. Ruder. 2018.*Universal Language Model Fine-tuning for Text Classification*. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146).
