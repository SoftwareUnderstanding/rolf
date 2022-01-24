cluster-preprocessing
=====================

This README explains the pre-processing performed to create the cluster lexicons that are used as features in the IXA pipes tools [http://ixa2.si.ehu.es/ixa-pipes]. So far we use the following three methods: Brown, Clark and Word2vec.

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Brown clusters](#brown)
3. [Clark clusters](#clark)
4. [Word2vec clusters](#word2vec)
5. [XML/HTML cleaning](#cleaning)

## OVERVIEW

We induce the following clustering types:

+ **Brown hierarchical word clustering algorithm**: [Brown, et al.: Class-Based n-gram Models of Natural Language.](http://acl.ldc.upenn.edu/J/J92/J92-4003.pdf)
  + Input: a sequence of words separated by whitespace with no punctuation. See [brown-input.txt](https://github.com/percyliang/brown-cluster/blob/master/input.txt) for an example.
  + Output: for each word type, its cluster. See [brown-output.txt](https://github.com/percyliang/brown-cluster/blob/master/output.txt) for an example.
  + In particular, each line is:
  ````shell
  <cluster represented as a bit string> <word> <number of times word occurs in input>
  ````
  + We use [Percy Liang's implementation](https://github.com/percyliang/brown-cluster) off-the-shelf.
  + [Liang: Semi-supervised learning for natural language processing.](http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf)

+ **Clark clustering**: [Alexander Clark (2003). Combining distributional and morphological information for part of speech induction](http://www.aclweb.org/anthology/E03-1009).
  + Input: one lowercased token per line, punctuation removed, sentences separated by two newlines. See [clark-input.txt](https://github.com/ragerri/cluster-preprocessing/blob/master/examples/clark-input.txt)
  + Output: for each word type, its cluster and a weight. See [clark-output.txt](https://github.com/ragerri/cluster-preprocessing/blob/master/examples/clark-output.txt)
  + Each line consists of
  ````shell
  <word> <cluster> <weight>
  ````
  + We use [Alexander Clark's implementation](https://github.com/ninjin/clark_pos_induction) off-the-shelf.

+ **Word2vec Skip-gram word embeddings clustered via K-Means**: [Mikolov et al. (2013). Efficient estimation of word representations in Vector Space.](http://arxiv.org/pdf/1301.3781.pdf)
  + Input: lowercased tokens separated by space, punctuation removed. See [word2vec-input.txt](https://github.com/ragerri/cluster-preprocessing/blob/master/examples/word2vec-input.txt)
  + Output: for each word type, its cluster. See [word2vec-output.txt](https://github.com/ragerri/cluster-preprocessing/blob/master/examples/word2vec-output.txt)
  + Each line consists of
  ````shell
  <word> <cluster>
  ````
  + We use [Word2vec implementation](https://code.google.com/archive/p/word2vec/) off-the-shelf.

## Brown

Let us assume that the source data is in plain text format (e.g., without html or xml tags, etc.), and that every document is in a directory called *corpus-directory*. Then the following steps are performed:

### Preclean corpus

+ Remove all sentences or paragraphs consisting of less than 90\% lowercase characters, as suggested by [Liang: Semi-supervised learning for natural language processing.](http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf).

This step is performed by using the following function in [ixa-pipe-convert](https://github.com/ragerri/ixa-pipe-convert):

````shell
java -jar ixa-pipe-convert-$version.jar --brownClean corpus-directory/
````

ixa-pipe-convert will create a *.clean* file for each file contained in the folder *corpus-directory*.

+ Move all *.clean* files into a new directory called, for example, *corpus-preclean*.

### Tokenize clean files to oneline format

+ Tokenize all the files in the folder to one line per sentence. This step is performed by using [ixa-pipe-tok](https://github.com/ixa-ehu/ixa-pipe-tok) in the following shell script:

````shell
./recursive-tok.sh $lang corpus-preclean
````
The tokenized version of each file in the directory *corpus-preclean* will be saved with a *.tok* suffix.

+ **cat to one large file**: all the tokenize files are concatenate it into a large huge file called *corpus-preclean.tok*.

````shell
cd corpus-preclean
cat *.tok > corpus-preclean.tok
````

### Format the corpus for Liang's implementation

+ Run the brown-clusters-preprocess.sh script like this to create the format required to induce Brown clusters using [Percy Liang's program](https://github.com/percyliang/brown-cluster).

````shell
./brown-clusters-preprocess.sh corpus-preclean.tok > corpus-preclean.tok.punct
````
### Induce Brown clusters:

````shell
brown-cluster/wcluster --text corpus-preclean.tok.punct --c 1000 --threads 8
````
This trains 1000 class Brown clusters using 8 threads in parallel.

## Clark

Let us assume that the source data is in plain text format (e.g., without html or xml tags, etc.), and that every document is in a directory called *corpus-directory*. Then the following steps are performed:

### Tokenize clean files to oneline format

+ Tokenize all the files in the folder to one line per sentence. This step is performed by using [ixa-pipe-tok](https://github.com/ixa-ehu/ixa-pipe-tok) in the following shell script:

````shell
./recursive-tok.sh $lang corpus-directory
````
The tokenized version of each file in the directory *corpus-directory* will be saved with a *.tok* suffix.

+ **cat to one large file**: all the tokenize files are concatenate it into a large huge file called *corpus.tok*.

````shell
cd corpus-directory
cat *.tok > corpus.tok
````

### Format the corpus

+ Run the clark-clusters-preprocess.sh script like this to create the format required to induce Clark clusters using [Clark's implementation](https://github.com/ninjin/clark_pos_induction).

````shell
./clark-clusters-preprocess.sh corpus.tok > corpus.tok.punct.lower
````

### Train Clark clusters:

To train 100 word clusters use the following command line:

````shell
cluster_neyessenmorph -s 5 -m 5 -i 10 corpus.tok.punct.lower - 100 > corpus.tok.punct.lower.100
````

## Word2vec

Assuming that the source data is in plain text format (e.g., without html or xml tags, etc.), and that every document is in a directory called *corpus-directory*. Then the following steps are performed:

### Tokenize clean files to oneline format

+ Tokenize all the files in the folder to one line per sentence. This step is performed by using [ixa-pipe-tok](https://github.com/ixa-ehu/ixa-pipe-tok) in the following shell script:

````shell
./recursive-tok.sh $lang corpus-directory
````
The tokenized version of each file in the directory *corpus-directory* will be saved with a *.tok* suffix.

+ **cat to one large file**: all the tokenize files are concatenate it into a large huge file called *corpus.tok*.

````shell
cd corpus-directory
cat *.tok > corpus.tok
````

### Format the corpus

+ Run the word2vec-clusters-preprocess.sh script like this to create the format required by [Word2vec](https://code.google.com/archive/p/word2vec/).

````shell
./word2vec-clusters-preprocess.sh corpus.tok > corpus-word2vec.txt
````

### Train K-means clusters on top of word2vec word embeddings

To train 400 class clusters using 8 threads in parallel we use the following command:

````shell
word2vec/word2vec -train corpus-word2vec.txt -output corpus-s50-w5.400 -cbow 0 -size 50 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 8 -classes 400
````

## Cleaning XML, HTML and other formats

There are many ways of cleaning XML, HTML and other metadata than often comes in corpora. As we will usually be processing very large amounts of texts, we do not pay too much attention to detail and crudely remove every tag using regular expressions. In the scripts directory there is a shell script that takes either a file as argument like this:

````shell
./xmltotext.sh file.html > file.txt
````
**NOTE that this script will replace your original files with a cleaned version of them**.

### Wikipedia

If you are interested in using the Wikipedia for your language, here you can find many Wikipedia dumps already extracted to XML which can be directly fed to the xmltotext.sh script:

[http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/]

If your language is not among them, we usually use the Wikipedia Extractor and then the xmltotext.sh script:

[http://medialab.di.unipi.it/wiki/Wikipedia_Extractor]

## Contact information

````shell
Rodrigo Agerri
IXA NLP Group
University of the Basque Country (UPV/EHU)
E-20018 Donostia-San Sebastián
rodrigo.agerri@ehu.eus
````
