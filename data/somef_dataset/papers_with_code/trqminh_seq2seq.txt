# seq2seq
## description
In this repo, I have written and refactored some code to mirror strings using the seq2seq model. For example, if the input is ABC, the ground-truth output is CBA.
## references
+ https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py  
+ https://arxiv.org/abs/1409.3215  
+ https://github.com/vanhuyz/udacity-dl/blob/master/6_lstm-Problem3.ipynb  
## requirements
CUDA 9.0 + Pytorch 1.0
## to run
ROOT = path/to/this/repo
- download the dataset
```python
cd ROOT
python dataset/download.py
```
- training:
```
python main.py
```
- inference:
```
python inference.py
```
## future works
attention, word embedding (glove or bert)
