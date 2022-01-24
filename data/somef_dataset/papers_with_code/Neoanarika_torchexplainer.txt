# torchexplainer : Axiomatic Attribution for NMT

This is a Pytorch implemementation of Axiomatic Attribution for Deep Networks specifically for NMT application. The underlying NMT model is from the PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017) by Yu-Hsiang Huang. The seq2seq impelementation is on the seq2seq branch. 

# Requirement
- python 3.4+
- pytorch 0.4.1+
- tqdm
- numpy


# Usage

## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz
```

### 1) Preprocess the data.
```bash
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low.pt
```

### 2) Train the model
```bash
python train.py -data data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding. 

### 3) Test the model
```bash
python translate.py -model trained.chkpt -vocab data/multi30k.atok.low.pt -src data/multi30k/test.en.atok -no_cuda
```

### 4) Attribute
```bash
python attribution.py -model trained.chkpt -data data/multi30k.atok.low.pt -out igs.pkl -no_cuda
```

To close all the matplotlib figures type
```
ps aux | grep python 
kill <process_id>
```

### 5) Debug
```bash
python attribution.py -model trained.chkpt -data data/multi30k.atok.low.pt -out igs.pkl -no_cuda -debug
```

# Results
![Alt Text](https://github.com/Neoanarika/torchexplainer/blob/master/translation.png)
Figure 1: Translating English to German, the brighter the square the more the model uses that word to come up with its corresponding translation. 
![Alt Text](https://github.com/Neoanarika/torchexplainer/blob/master/tgt_IG.png)
Figure 2: Translating English to German, this time visualising the target sequence instead of the outpt sequence 

# Acknowledgement
- https://github.com/jadore801120/attention-is-all-you-need-pytorch by Yu-Hsiang Huang
