# Master's Project: Conversation Structure Analysis + Visualisation using Podcast Transcripts

In this MSci project we are looking at the structure of human-human conversation under 2 different lenses: as a macroscopic trajectory through a topic space (built from the components of word embeddings) and as a string of microscopic dialogue acts. From the topic space analysis we hope to build a signature graphical representation of the transcript which visualises the evolution of topics discussed, and portrays key information about the given conversation. From the dialogue analysis we hope to answer the question 'What makes Conversation Interesting?'.

*Key libraries used: sklearn, torch, gensim, nltk, spacy, scipy, pandas, networkx*

# Key Methods Used

*1) Keyword Extraction*
- Keywords and phrases, using PKE implementation of TopicRank.
- Nouns, using spacy POS tagger with the en_core_web_sm pretrained statistical model. 
- Bigrams and Trigrams, using NLTK implementation of Collocation Finder.

*2) Topic Space Construction*
- Word2Vec implementation with GoogleNews-vectors-negative300 pretrained word embeddings.
- FastText implementation with the cc.en.300.bin pretrained model.

*3) Transcript Segmentation*
- Infersent implementation with arbitrary cosine similarity cutoff between the embeddings of consecutive sentences.
- SliceCast implementation.
- Latent Dirichlet Allocation (*BayesSeg*)
- Utterance-Embedding Clustering

*4) Dialogue Act Classification*
- The [Bi-LSTM-CRF](https://arxiv.org/abs/1709.04250) Model by Kumar et al. 

## Useful Links
Embedding techniques used:
* Sentence embeddings with [InferSent](https://github.com/facebookresearch/InferSent) developed by Facebook Research for utterance-level analysis, from the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364). 
* Word embeddings with [Word2Vec](https://arxiv.org/abs/1301.3781) for EDU-level analysis, uing [this](https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/) Word2Vec model pretrianed by Google.
* Word embeddings with [FastText](https://github.com/facebookresearch/fastText) based on the paper [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)for EDU-level analysis, using the [cc.en.300.bin](https://fasttext.cc/docs/en/crawl-vectors.html) model.

Segmentation methods used: 
* [SliceCast](https://github.com/bmmidei/SliceCast) implementation from [Neural Text Segmentation on Podcast Transcripts](https://github.com/bmmidei/SliceCast/blob/master/Neural_Text_Segmentation_on_Podcast_Transcripts.pdf).
* [Infersent](https://github.com/facebookresearch/InferSent) sentence embeddings paired with a variable cosine similarity cutoff. Graphical method inspired by paper [Minimum_Cut_Model_for_Spoken_Lecture_Segmentation](https://www.researchgate.net/publication/220873934_Minimum_Cut_Model_for_Spoken_Lecture_Segmentation).


# Discussion Trees: Visualising Conversation Structure and Topic Evolution
This part of the project investigates how one can best visualise the evolution of ideas and nature/flow of conversation from a given podcast transcript. We introduce Discussion Trees: graphics designed to offer a birds-eye-view of conversation transcripts in a way which aims to uncover high-level trends in human communication patterns. 

As output graphics from two podcast transcripts are structured according to the same set of rules, they can be directly visually compared and in doing so highlight differences in conversation structure and topical evolution. When applied to podcast transcripts, we propose Discussion Tree as a method of podcast hosts gaining insight into their unique interview styles, as a way to evaluate why one episode found more success than another, and also as means of transcript data navigation. Discussion Trees for the interviews of a) Jack Dorsey and b) Elon Musk by Joe Rogan on the Joe Rogan podcast. 

<img src = "Screenshots/Full_Trees.png" width ="600" />

Below on the left Discussion Tree of the first 350 utterances in the Joe Rogan interview of Jack Dorsey, with only the most popular topics – in terms of usage throughout the whole transcript – annotated on their respective stacks, and branch numbers annotated on the green leaves. A Detailed Discussion Tree for the topic of *Twitter* is given on the right. All branches of conversation originating on this topic are plotted, as well as conversation details given in the green label.

<img src = "Screenshots/First_350_Jack_Trimmed.png" width ="300" /> <img src = "Screenshots/Twitter_Jack_Trimmed.png" width ="300" />


Finally, the Discussion Trees for nine episodes from the ‘Heavy Topics’ podcast, available on Spotify, are given below to illustrate how they can be used as visual fingerprints for episodes of a given podcast.

<img src = "Screenshots/DTs_as_fingerprints.png" width ="400" />




