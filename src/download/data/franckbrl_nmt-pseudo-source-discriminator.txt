NMT-PSEUDO-SOURCE-DISCRIMINATOR
-------------------------------

Neural Machine Translation with Pseudo-Source Discriminator.

This implementation is based on the [Theano version](https://github.com/EdinburghNLP/nematus/tree/theano) of [Nematus](https://github.com/EdinburghNLP/nematus).

The Machine Translation model is trained on natural parallel data (the source is a human translation of the target), as well as pseudo-parallel data (the source is an automatic translation or a copy of the target). These two data types have a separate encoder and are used in a [Generative Adversarial Network](https://arxiv.org/abs/1406.2661) scenario. The output of both encoders is fed to a discriminator that is optimized to distinguish the encoding of natural and pseudo sources. The discriminator feedback is used to optimize the pseudo-source encoder.

INSTALLATION
------------

See the [Nematus repository](https://github.com/EdinburghNLP/nematus/tree/theano#installation).

USAGE INSTRUCTIONS
------------------

Execute `nmt.py` to train a model.

#### Additional arguments

The arguments are the same as in [Nematus](https://github.com/EdinburghNLP/nematus/tree/theano#usage-instructions), augmented with the following:


| parameter            | description |
|---                   |--- |
| --pseudo_data |  parallel training corpus with pseudo-source |
| --pretrain_pseudo_src | pretrain pseudo source encoder before NMT training starts |
| --generator_start_uidx | update number to start generator training (default: 10000) |
| --nmt_start_uidx | update number to start NMT training (default: 10000) |
| --pseudo_src_noise | introduce noise in pseudo source (drop words and make permutations) |
| --d_lrate | learning rate for pseudo-source discriminator (default: 1e-05) |
| --g_lrate | learning rate for pseudo-source generator (default: 1e-05) |


#### Inference

Inference is run just like in [Nematus](https://github.com/EdinburghNLP/nematus/tree/theano#nematustranslatepy--use-an-existing-model-to-translate-a-source-text).

PUBLICATIONS
------------

Franck Burlot and François Yvon, Using Monolingual Data in Neural Machine Translation: a Systematic Study. In Proceedings of the Third Conference on Machine Translation (WMT’18). Association for Computational Linguistics, Brussels, Belgium, 2018.
