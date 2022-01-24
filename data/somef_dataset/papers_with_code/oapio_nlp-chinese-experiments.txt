# Chinese embeddings and segmentation research

In this repo are code and notes related to experimentation with chinese embeddings. We would like to use embeddings of chinese words as input to our disambiguation and involvement models.
However, since chinese is not naturally segmented language (there are no spaces) this is problematic.
For any questions about this repo, contact Jan Bogar.

This README is organized as follows:
1. Papers and other sources I read, with notes
2. experiment with segmentation
3. Conclusion from the experiment, where to go next
4. For the future: where to get data

## Papers
### With segmentation
#### Fastext
 - https://arxiv.org/pdf/1607.04606.pdf
 - Subword information enhanced embeddings
 - “We used the Stanford word segmenter for Chinese”

#### Primer (just a blog):
 - https://primer.ai/blog/Chinese-Word-Vectors/
 - Segmentation with https://github.com/fxsjy/jieba

#### chinese NER
 - might be usefull in the future, otherwise useless
 - http://aclweb.org/anthology/D15-1064

#### Character enhanced chinese embeddings
 - before fasttext (2015)
 - http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf
 - No mention of segmentation, but they obviously used it 
 - https://github.com/Leonard-Xu/CWE

#### Cw2vec
 - http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf
 - 2018 paper, uses information from strokes in the characters, cool paper
 - Segmentation with https://github.com/NLPchina/ansj_seg
 - Comparisions with other methods included, this cw2vec seems best, then there are https://github.com/HKUST-KnowComp/JWE and http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf or glove (fasttext not included)

### Without segmentation

#### Sembei
 - http://aclweb.org/anthology/D17-1080 
 - C++ implementation at https://github.com/oshikiri/w2v-sembei , seems buggy? 
 
#### Combination of Sembei and Fasttext
- https://arxiv.org/pdf/1809.00918.pdf
- New paper, no implementation anywhere

#### Ngram2vec
 - https://github.com/zhezhaoa/ngram2vec   **<--this is cool, might be useable out of the box**
 - http://www.aclweb.org/anthology/D17-1023 , implemented
 - **About ngrams of words, so might be turned into ngrams of characters**
 - Contains some chinese analogy datasets


## Experiments with segmentation

**Idea:** preprocess whole index with segmentation? How much would it cost? How long would it take?

Zhtw factiva indexed has 120 GB total (there is also Zhcn factiva).
Jieba segmenter (https://github.com/fxsjy/jieba) claims 1.5 MB/s, which means whole indexed zhtw factiva would be segmented in a single day on one machine.

Jieba also has java and rust implementations, which might be even faster:
 - Java implementation of jieba: https://github.com/huaban/jieba-analysis/blob/master/src/main/java/com/huaban/analysis/jieba/JiebaSegmenter.java
 - rust implementation of jieba:  https://github.com/messense/jieba-rs

### Accuracy of segmentation
In this notebook I compared accuracy of jieba segmenter and Stanford NLP segmenter against annotated dataset:
https://github.com/oapio/nlp-chinese-experiments/blob/master/segmenters%20test.ipynb

Results:
average boundaries in a sentence:  26.3
average jieba wrong boundaries: number: 4.38 ,   ratio: 0.167
average stanford wrong boundaries: number:  3.49 ,   ratio: 0.133

Dataset size- 46364 sentences
Number of sentences without error: 4622 for jieba, 4843 for stanfordNLP

### word length distribution
https://github.com/oapio/nlp-chinese-experiments/blob/master/word%20lengths.ipynb

Results: 95 % of words has three characters or less, 90 % characters has 2 characters or less

## Conclusion
Majority of approaches to chinese NLP (including embeddings) assumes segmentation of chinese sentences as a first step. In light of that, I researched two tools for chinese language segmentation: Stanford NLP Segmenter and Jieba .

On my human annotated dataset, only about 10% of sentences are segmented without any error. Even if the segmentation rules are not clear, it is alarmingly low success ratio. Accuracy on individual boundaries is for both jieba and stanford sentencer are below 90 %.

Jieba would be capable of segmenting  whole factiva in a matter of 1-2 days on a single machine.
There are three possible ways to use chinese embeddings in our pipeline:

**Segmentation  + embeddings with fasttext**
 - Segmentation could simplify other  NLP tasks
 - This would keep chinese pipeline similar to english pipeline, which would simplify development
 - Fasttext embeddings for chinese words are freely available and would be easy to use . Also training the embeddings on our own  datasets would be relatively easy.
 - Fasttext might also be partially immune to effects of erroneous segmentation, since it uses subword information for learning of embeddings, and therefore might assign approximately correct vector also to word that is incomplete or has some characters added. This is however untested hypothesis.

**Segmentation free approach**
 - segmentation is unreliable and introduces another source of error early on in the pipeline.
 - For segmentation-free approaches, implementations are few. Training of our own embeddings would require a lot more effort (about a week of research and coding for working prototype at best, unless ngram2vec proves viable option).
 - The pipeline would be simplified (but would also diverge from english pipeline) and precision could be potentially higher.

**Use fasttext on unsegmented text ( e.g. use all ngrams in the text instead of words)**
 - Very easy
 - Likely worse than both of the above

My recommendation is to use segmentation+fasttext and fasttext on unsegmented text as as a reasonable first step.
As a second iteration I would focuse on Ngram2Vec. I would trick it to treat each character as separate word. Since ngram2vec is supposed to learn embeddings for ngrams of words, it would instead learn embeddigns of ngrams of characters. Since it's likely to work almost out of the box, it is a reasonable segmentation free approach and we would get all utilities shipped with it for free.

## For the future: where to get data
For any future chinese embeddigns research, we will need huge corpus of raw chinese text.
Luckily, we have whole factiva clone in jsonl format in google storage (also indexed in elasticsearch).

Data is described in the beggining of this document: https://docs.google.com/document/d/1j_5AYKNEM0tbRgixkmM1OzGjLbUHWIB52kYPRzCdVbY/edit#heading=h.xtuqoz5uvrzr

Link to the data is: https://console.cloud.google.com/storage/browser/factiva-snapshots-processed/snapshot-full?project=merlon-182319&authuser=0&pli=1&angularJsUrl=%2Fstorage%2Fbrowser%2Ffactiva-snapshots-processed%3Fproject%3Dmerlon-182319%26authuser%3D1%26pli%3D1

To download the data and other operations, I strongly reccomend use of gsutil tool.
If you will train embeddings in google cloud, you don't have to download the data, so just download one month or one year for experiments.
It is really just a bunch of jsonl files with one article per line, sorted into directories by language, year and month.

Person of contact for the data is Michal Nanasi.
