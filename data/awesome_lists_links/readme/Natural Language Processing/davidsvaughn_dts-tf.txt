# dts-tf
Deep Text Scoring in TensorFlow

- this code implements either a _flat_ or a 2 level _hierarchical_ RNN model <br/> (see paper below: _Hierarchical Attention Networks..._) 
- can operate either on _character_ sequences or _word_ sequences... pretrained embeddings inluded for both
- everything is controlled through configuration (*.conf) file, which is set at top of train.py: <br/> i.e. `config_file = 'config/han.conf'`
- several examples, including config files and output logs, are given in `chkpt` folder

## Get data from spshare
- copy files from: `//spshare/users/dvaughn/dts-tf/embeddings` to `./embeddings`
- copy files from: `//spshare/users/dvaughn/dts-tf/data` to  `./data`
- theres more data in: `//spshare/users/dvaughn/data` 
    - see the `readme.txt` file and `./scripts` subfolder for more info


## Run
```
cd python
python -u train.py | tee log.txt
```

## References

### Code

- [tf-lstm-char-cnn](https://github.com/mkroutikov/tf-lstm-char-cnn) ...source of character embedding model
- [BERT: PyTorch Conversion](https://github.com/huggingface/pytorch-pretrained-BERT) ...WordPieces implemened here
- [BERT: Original TensorFlow Code](https://github.com/google-research/bert)

### Papers

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) ...WordPieces
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) ...WordPieces
- [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474) ... rhn unit
- [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) ...char embedding model