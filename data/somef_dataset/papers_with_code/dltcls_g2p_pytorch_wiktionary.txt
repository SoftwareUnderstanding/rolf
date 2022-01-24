# Sequence-to-Sequence with PyTorch

Sequence-to-Sequence - G2P, P2G or P2P

**Language: German**

## Project structure
```
│   create_toy_dataset.py           ### Creates toy datasets
│   de_wiktionary_main.py           ### Parses wiktioanry xml file        
│   pytorch_main.py                 ### Trains the model
│   README.md
├───data                            ### Toy datasets
│   │   g2p_toy_wiki_de-de_3.csv
│   │   p2g_toy_wiki_de-de_3.csv
│   │   p2p_toy_wiki_de-de_3.csv
│   │
│   ├───preprocessed                ### Preprocessed wiktionary dump file
│   │       wiki_corpus_de.csv
│   │
│   └───wiki
│           dewiktionary-latest-pages-articles.xml  ### Extracted wiktionary dump file
│
├───results
│   └───pytorch
│       ├───g2p 					### training mode
│       │   └───3					### sequence length
│       │       └───2019-06-20-13-12-17	### execution date 
│       │               experiment.log		### Experiment infos (data, model configuration, experiment configuration)
│       │               results.log			### Logging during training (loss, PER, test translations)
│       │               g2p_3.pth			### Model stored (<mode>_<seq_len>.pth
│       │ 
```

## Required packages
- PyTorch
- Torchtext
- Levenshtein
- pandas
- numpy
- mwxml (Wiktionary parser)


## Data

The model is trained on words and their IPA transcriptions as it can be found in the Wikimedia **Wiktionary** for the German language.
The Wiktionary dump can be downloaded at this link: https://dumps.wikimedia.org/dewiktionary/latest/dewiktionary-latest-pages-articles.xml.bz2

Once downloaded, the dump should be extracted to the directory: **data/wiki**.

The preprocessed dump file is stored as `csv` file in the **preprocessed** directory (`wiki_corpus_de.csv`).

The script for preprocessing the wiktionary dump is: `de_wiktionary_main.py`.


## Toy Data 

After having preprocessed the Wiktionary dump file, it is possible to create toy datasets.
This is done with the script `create_toy_dataset.py`.   

The program accepts following arguments:
- `seq_len`: how many units (words, phoneme-words) should be considered
- `mode`: accepted are *g2p*, *p2g*, *p2p*
- `samples`: How many examples should be created. Default 0 (same number as in the wiktioanry file)

# Model

The model is a simple Encoder-Decoder-Architecture.
The implementation is based on this tutorial: https://fehiepsi.github.io/blog/grapheme-to-phoneme/, which builds a G2P model based on the CMU dictionary.
Here the model works on real words and their IPA transcriptions as in Wikipedia.

Both Encoder and Decoder have one `LSTMCell` which processes the input.
The model works with the `Attention` mechanism. For further details please check the tutorial.

A `Beam` search is performed at test time.

# Training modes

The model works at **char-level**, so it is basically a Char2Char model.
Input sequences are reversed (see: https://arxiv.org/pdf/1409.3215.pdf).


## G2P

This trains the model in the grapheme-to-phoneme mode.
Please provide training `mode` as `g2p` and a file beginning with `g2p_`. 


## P2G
This trains the model in the phoneme-to-grapheme mode.
Please provide training `mode` as `p2g` and a file beginning with `p2g_`. 

## P2P - Word boundaries detection
This trains the model in the phoneme-to-phoneme mode.
Please provide training `mode` as `p2p` and a file beginning with `p2p_`.

In this mode, the model should learn the word boundaries given a not splitted input sequence:
Example:

Input: uɛsɡʁosbʏʁɡɐlɪçmʁepʁivatiziɐtn	
Target: uɛs ɡʁosbʏʁɡɐlɪçm ʁepʁivatiziɐtn 

### TODO
- Model: Setup the model to be multilayered or to work bidirectionally
- Training: Try with more parameters

### SOURCES
Tutorial: https://fehiepsi.github.io/blog/grapheme-to-phoneme/