# Neural Machine Translation Project Module

* GloVe http://www.aclweb.org/anthology/D14-1162
* Evaluation methods http://www.aclweb.org/anthology/D15-1036
* Intrinsic Evaluation http://www.aclweb.org/anthology/W16-2507
* preprocessing steps and hyperparameter settings http://www.aclweb.org/anthology/Q15-1016
* WMT 2017 Translation Task http://www.statmt.org/wmt17/translation-task.html
* Bilingual Data used http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/
* Additional monolingual Data used http://www.statmt.org/wmt14/training-monolingual-news-crawl/

----------------------------------------------------------------------------------------------

## Training Steps

Clone this repository in the desired place:

    git clone https://github.com/kinimod23/NMT_Project.git
    cd ~/NMT_Project/NMT_environment/shell_scripts

Set up the NMT environment:

    bash sockeye_wmt_env.sh

Preprocess the data used:

    bash sockeye_wmt_prep.sh

Pre-train glove embeddings:

    cd ~/NMT_Project/NLR_pre-training/glove

Download and install glove components:

    git init .
    git remote add -t \* -f origin http://github.com/stanfordnlp/glove
    git checkout master
    make

Train glove embeddings with previously generated BPE training data:

    # for source
    bash glove_small.training.sh ~/NMT_Project/NMT_environment/data/train.BPE.en
    # for target
    bash glove_small.training.sh ~/NMT_Project/NMT_environment/data/train.BPE.de

Initialize pre-trained embedding matrix for final NMT training:

    cd ~/NMT_Project/NMT_environment/shell_scripts
    bash sockeye_wmt_create.small.embs.sh

Final NMT training - Baseline (with insulated Embeddings):

    bash sockeye_wmt_train_basel.sh

Final NMT training - Experiment (with pre-trained Embeddings on small Corpus):

    bash sockeye_wmt_train_small.prembs.sh model_wmt17_small.glove


## Use more data to pre-train glove embeddings

    cd ~/NMT_Project/NMT_environment/shell_scripts
    bash sockeye_wmt_prep_add.data

Train glove embeddings with previously generated additional BPE training data:

    cd ~/NMT_Project/NLR_pre-training/glove
    # for source
    bash glove_large.training.sh ~/NMT_Project/NMT_environment/data/pre-train_data/pre-train.BPE.en
    # for target
    bash glove_large.training.sh ~/NMT_Project/NMT_environment/data/pre-train_data/pre-train.BPE.de

Initialize pre-trained embedding matrix for final NMT training:

    cd ~/NMT_Project/NMT_environment/shell_scripts
    bash sockeye_wmt_create.large.embs.sh

Final NMT training - Experiment (with pre-trained Embeddings on large Corpus):

    bash sockeye_wmt_train_large.prembs.sh model_wmt17_large.glove


## Evaluation Steps

Using test data for Evaluation

    cd ~/NMT_Project/NMT_environment/shell_scripts
    # Evaluation of baseline model
    bash sockeye_wmt_eval.sh model_wmt17_basel
    # Evaluation of glove model pre-trained on small data
    bash sockeye_wmt_eval.sh model_wmt17_small.glove
    # Evaluation of glove model pre-trained on large data
    bash sockeye_wmt_eval.sh model_wmt17_large.glove

\
\
Doing a recheck if the initially used vectors of the sockeye-nmt-system are actually conform with the pre-trained vectors (and not Zero as being the usual "sockeye way")

[1] extract initial sockeye-nmt-system's embedding vectors

    # for small
    bash sockeye_wmt_prembs.recheck.sh model_wmt17_small.glove && exit
    # for large
    bash sockeye_wmt_prembs.recheck.sh model_wmt17_large.glove && exit

[2] on local machine

    mkdir ~/Desktop/recheck_embs
    cd ~/Desktop/recheck_embs
    wget https://raw.githubusercontent.com/kinimod23/NMT_Project/master/NMT_environment/tools/recheck_embs.sh
    wget https://raw.githubusercontent.com/kinimod23/NMT_Project/master/NMT_environment/tools/np_transf.py
    wget https://raw.githubusercontent.com/kinimod23/NMT_Project/master/NMT_environment/tools/recheck_initvecs.py
    wget https://raw.githubusercontent.com/kinimod23/NMT_Project/master/NMT_environment/tools/recheck_cosines.py

[3] download and transform vectors for rechecking

    # for baseline
    bash recheck_embs.sh model_wmt17_basel
    # for large glove
    bash recheck_embs.sh model_wmt17_large.glove


[4] use script to compare pre-trained vs. initially used vectors

    python recheck_initvecs.py large.src_init.txt large.glove.en.txt
    python recheck_initvecs.py large.trg_init.txt large.glove.de.txt

    # the output is a print statement telling if all glove embeddings are found in sockeye's embedding layer and if not how many

\
\
Doing another recheck of how much embeddings change from params.00000 to params.best

    # for baseline
    python recheck_cosines.py basel.src_init.txt best.basel.src_init.txt 
    python recheck_cosines.py basel.trg_init.txt best.basel.trg_init.txt
    # for large glove
    python recheck_cosines.py large.src_init.txt best.large.src_init.txt
    python recheck_cosines.py large.trg_init.txt best.large.trg_init.txt

    # the output is an image file in the form of a histogram showing the frequency distribution on cosine distances between 0-1
&nbsp;

## Significance testing

    cd ~/NMT_Project/Signifikanztests
    # activate python environment, download test tool & copy required data
    bash signi_env.sh

Execute significance test with:\
<sub>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arg1 = *give a name for the model*\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arg2 = gold standard\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arg3 = translated test sentences of System 1\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arg4 = translated test sentences of System 2
</sub>

    bash signi_test.sh basel.sglove test.gold.de test.transl.basel.de test.transl.small.glove.de

    bash signi_test.sh basel.lglove test.gold.de test.transl.basel.de test.transl.large.glove.de

----------------------------------------------------------------------------------------------

## ToDo
* finish, smooth and polish seminar paper

----------------------------------------------------------------------------------------
## What I have done
* evaluated how much embeddings change from params.00000 to params.best using a histogram on cosine distances

* pre-trained embeddings on additional/different data

* significance testing

* evaluation of pre-trained vs. initial sockeye-nmt-system's embedding vectors using a script calculating intersections

* evaluation via BLEU score

* sockeye NMT model trained with glove embeddings on the wmt'17 corpus

* glove embeddings trained on BPE-Units

* successfully run a NMT toy model using sockeye

* implemented glove, zalando, elmo and paragraph-vector NLRs
	* for all there are still some challenges to overcome except of glove
	
* written Exposé with goals of this project
    * Literature survey on Research Questions

---------------------------------------------------------------------------------------------------

### Project Organisation

#### A short memorable project title.
An Evaluation of different Natural Language Representations by using an identical Neural Machine Translation Network

#### What is the problem you want to address? How do you determine whether you have solved it?
To categorise distinct approaches (character/word/sentence/thought input) for generating word embeddings.
By using a translation task (from English to German), it's clear to see which approach performs best.

Research Questions:
a) Which is the best lexical input (character, word, sentence, thought) to generate language representations for a translation task?
b) Which is the best Language Model (bi-directional, one-directional, etc.) to use for generating language representations applied to a translation task?

#### How is this going to translate into a computational linguistics problem?
Natural language Representations (NLRs) might ignore key features of distributional semantics! A new NLR model is typically evaluated across several tasks, and is considered an improvement if it achieves better accuracy than its predecessors. However, different applications rely on different aspects of word embeddings, and good performance in one application does not necessarily imply equally good performance on another.

#### Which data are you planning to use?
WMT 2017 Translation Task http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/

------------------------------------------------------------------------------------------
