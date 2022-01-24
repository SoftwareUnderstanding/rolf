# seq2seq
Attention-based sequence to sequence learning

## Dependencies

* [TensorFlow 1.2+ for Python 3](https://www.tensorflow.org/get_started/os_setup.html)
* YAML and Matplotlib modules for Python 3: `sudo apt-get install python3-yaml python3-matplotlib`
* A recent NVIDIA GPU

## How to use


Train a model (CONFIG is a YAML configuration file, such as `config/default.yaml`):

    ./seq2seq.sh CONFIG --train -v 


Translate text using an existing model:

    ./seq2seq.sh CONFIG --decode FILE_TO_TRANSLATE --output OUTPUT_FILE
or for interactive decoding:

    ./seq2seq.sh CONFIG --decode

#### Example English&rarr;French model
This is the same model and dataset as [Bahdanau et al. 2015](https://arxiv.org/abs/1409.0473).

    config/WMT14/download.sh    # download WMT14 data into raw_data/WMT14
    config/WMT14/prepare.sh     # preprocess the data, and copy the files to data/WMT14
    ./seq2seq.sh config/WMT14/baseline.yaml --train -v   # train a baseline model on this data

You should get similar BLEU scores as these (our model was trained on a single Titan X I for about 4 days).

| Dev   | Test  | +beam | Steps | Time |
|:-----:|:-----:|:-----:|:-----:|:----:|
| 25.04 | 28.64 | 29.22 | 240k  | 60h  |
| 25.25 | 28.67 | 29.28 | 330k  | 80h  |

Download this model [here](https://drive.google.com/file/d/1Qe4yZTYSTF-mlRlP_NTFGwXgacZnBwdp/view?usp=sharing). To use this model, just extract the archive into the `seq2seq/models` folder, and run:

     ./seq2seq.sh models/WMT14/config.yaml --decode -v

#### Example German&rarr;English model
This is the same dataset as [Ranzato et al. 2015](https://arxiv.org/abs/1511.06732).

    config/IWSLT14/prepare.sh
    ./seq2seq.sh config/IWSLT14/baseline.yaml --train -v

| Dev   | Test  | +beam | Steps |
|:-----:|:-----:|:-----:|:-----:|
| 28.32 | 25.33 | 26.74 | 44k   |

The model is available for download [here](https://drive.google.com/file/d/1qCL3ZRxZ13fC45f74Nt6qiQ8tVAYFF9H/view?usp=sharing).

## Audio pre-processing
If you want to use the toolkit for Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST), then you'll need to pre-process your audio files accordingly.
This [README](https://github.com/eske/seq2seq/tree/master/config/BTEC) details how it can be done. You'll need to install the **Yaafe** library, and use `scripts/speech/extract-audio-features.py` to extract MFCCs from a set of wav files.

## Features
* **YAML configuration files**
* **Beam-search decoder**
* **Ensemble decoding**
* **Multiple encoders**
* **Hierarchical encoder**
* **Bidirectional encoder**
* **Local attention model**
* **Convolutional attention model**
* **Detailed logging**
* **Periodic BLEU evaluation**
* **Periodic checkpoints**
* **Multi-task training:** train on several tasks at once (e.g. French->English and German->English MT)
* **Subwords training and decoding**
* **Input binary features instead of text**
* **Pre-processing script:** we provide a fully-featured Python script for data pre-processing (vocabulary creation, lowercasing, tokenizing, splitting, etc.)
* **Dynamic RNNs:** we use symbolic loops instead of statically unrolled RNNs. This means that we don't mean to manually configure bucket sizes, and that model creation is much faster.

## Credits

* This project is based on [TensorFlow's reference implementation](https://www.tensorflow.org/tutorials/seq2seq)
* We include some of the pre-processing scripts from [Moses](http://www.statmt.org/moses/)
* The scripts for subword units come from [github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)
