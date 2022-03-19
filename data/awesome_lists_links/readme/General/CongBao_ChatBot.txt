# ChatBot
A simple implementation of RNN-based chat bot.

Applied attention mechanism, mixed teacher forcing and greedy approach in training,
and beam search in decoding. Deal with OOV problem in pre-trained word embeddings.

## Usage

#### Train Models

Simplest (dialog text and word embedding are required)

    python train.py --text ~/data/dialog_text.txt --embed ~/embeds/glove.txt

Specify hyperparameters (learning rate, batch size, epochs, teacher forcing ratio)

    python train.py [...] -r 0.01 -b 32 -e 50 --tfr 0.6

Specify file path (checkpoint directory)

    python train.py [...] --ckpt ~/test/checkpoints/
    
Training on CPU

    python train.py [...] --cpu-only
    

#### Chat with Trained Model

Simplest (dialog text is required)

    python chat.py --text ~/data/dialog_text.txt
    
Specify hyperparameters (mode, beam size)

    python chat.py [...] -m beam -k 5
    python chat.py [...] -m greedy
    
Specify file path

    python train.py [...] --ckpt ~/test/checkpoints/
    
Processing on CPU

    python train.py [...] --cpu-only
    
## References

Main Reference (seq2seq on chat bot)

    @article{DBLP:journals/corr/VinyalsL15,
      author    = {Oriol Vinyals and
                   Quoc V. Le},
      title     = {A Neural Conversational Model},
      journal   = {CoRR},
      volume    = {abs/1506.05869},
      year      = {2015},
      url       = {http://arxiv.org/abs/1506.05869},
      archivePrefix = {arXiv},
      eprint    = {1506.05869},
      timestamp = {Mon, 13 Aug 2018 16:48:58 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/VinyalsL15},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }

Long Short-Term Memory

    @article{Hochreiter:1997:LSM:1246443.1246450,
      author = {Hochreiter, Sepp and Schmidhuber, J\"{u}rgen},
      title = {Long Short-Term Memory},
      journal = {Neural Comput.},
      issue_date = {November 15, 1997},
      volume = {9},
      number = {8},
      month = nov,
      year = {1997},
      issn = {0899-7667},
      pages = {1735--1780},
      numpages = {46},
      url = {http://dx.doi.org/10.1162/neco.1997.9.8.1735},
      doi = {10.1162/neco.1997.9.8.1735},
      acmid = {1246450},
      publisher = {MIT Press},
      address = {Cambridge, MA, USA},
    }

    @article{DBLP:journals/corr/Graves13,
      author    = {Alex Graves},
      title     = {Generating Sequences With Recurrent Neural Networks},
      journal   = {CoRR},
      volume    = {abs/1308.0850},
      year      = {2013},
      url       = {http://arxiv.org/abs/1308.0850},
      archivePrefix = {arXiv},
      eprint    = {1308.0850},
      timestamp = {Mon, 13 Aug 2018 16:47:21 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/Graves13},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
Seq2seq

    @article{DBLP:journals/corr/ChoMGBSB14,
      author    = {Kyunghyun Cho and
                   Bart van Merrienboer and
                   {\c{C}}aglar G{\"{u}}l{\c{c}}ehre and
                   Fethi Bougares and
                   Holger Schwenk and
                   Yoshua Bengio},
      title     = {Learning Phrase Representations using {RNN} Encoder-Decoder for Statistical
                   Machine Translation},
      journal   = {CoRR},
      volume    = {abs/1406.1078},
      year      = {2014},
      url       = {http://arxiv.org/abs/1406.1078},
      archivePrefix = {arXiv},
      eprint    = {1406.1078},
      timestamp = {Mon, 13 Aug 2018 16:46:44 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/ChoMGBSB14},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
Attention

    @article{DBLP:journals/corr/LuongPM15,
      author    = {Minh{-}Thang Luong and
                   Hieu Pham and
                   Christopher D. Manning},
      title     = {Effective Approaches to Attention-based Neural Machine Translation},
      journal   = {CoRR},
      volume    = {abs/1508.04025},
      year      = {2015},
      url       = {http://arxiv.org/abs/1508.04025},
      archivePrefix = {arXiv},
      eprint    = {1508.04025},
      timestamp = {Mon, 13 Aug 2018 16:46:14 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/LuongPM15},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
Beam Search

    @article{DBLP:journals/corr/FreitagA17,
      author    = {Markus Freitag and
                   Yaser Al{-}Onaizan},
      title     = {Beam Search Strategies for Neural Machine Translation},
      journal   = {CoRR},
      volume    = {abs/1702.01806},
      year      = {2017},
      url       = {http://arxiv.org/abs/1702.01806},
      archivePrefix = {arXiv},
      eprint    = {1702.01806},
      timestamp = {Mon, 13 Aug 2018 16:49:02 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/FreitagA17},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
Word2vec & Negative Sampling

    @article{DBLP:journals/corr/MikolovSCCD13,
      author    = {Tomas Mikolov and
                   Ilya Sutskever and
                   Kai Chen and
                   Greg Corrado and
                   Jeffrey Dean},
      title     = {Distributed Representations of Words and Phrases and their Compositionality},
      journal   = {CoRR},
      volume    = {abs/1310.4546},
      year      = {2013},
      url       = {http://arxiv.org/abs/1310.4546},
      archivePrefix = {arXiv},
      eprint    = {1310.4546},
      timestamp = {Mon, 13 Aug 2018 16:47:09 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/MikolovSCCD13},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }