# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.

**BPE related parts are not yet fully tested.**


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Usage

# 라이브러리 설치
```bash
pip install msgpack==1.0.2
pip install msgpack-numpy==0.4.7.1
pip install spacy==2.3.5
pip install torchtext==0.4
```

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```


### 2) Train the model
수정 전
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

수정 후
```bash
python train.py -data_pkl m30k_deen_shr.pkl -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model
수정 전
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

수정 후
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model ./output/model.chkpt -output prediction.txt
```

### 4) calculate bleu
```bash
python bleu.py --reference ./.data/multi30k/test2016.en --candidate prediction.txt
```

# Performance
## Training

<p align="center">
<img src="https://i.imgur.com/S2EVtJx.png" width="400">
<img src="https://i.imgur.com/IZQmUKO.png" width="400">
</p>

- Parameter settings:
  - batch size 256 
  - warmup step 4000 
  - epoch 200 
  - lr_mul 0.5
  - label smoothing 
  - do not apply BPE and shared vocabulary
  - target embedding / pre-softmax linear layer weight sharing. 
 
  
## Testing 
(train.py에서 layer, head값 변경)

(before)

de -> en

layer 6, head 8
- (training) ppl : 6.85121, acu : 85.685 %
- (validation) ppl : 13.92095, acu: 62.202 %
- (test) bleu : 0.37815

layer 12, head 12
- (training) ppl : 8.18509, acu : 80.748 %
- (validation) ppl : 28.42237, acu: 50.873 %
- (test) bleu : 0.27540

layer 12, head 8
- (test) bleu : 0.28398

layer 8, head 12
- (training) ppl : 6.94947, acu : 85.097 %
- (Validation) ppl : 30.94104, acu : 51.773 %
- (test) bleu : 0.28535

layer 8, head 8
- (test) bleu : 0.29825

wmt 14 dataset (preprocess.py에서 multi30k -> wmt14로 변경)

en -> de

layer 6, head 8
- (training) ppl : 6.66036, acu : 86.251 %
- (validation) ppl : 14.95462, acu: 59.245 %
- (test) bleu : 0.20529



(after pull request merge)

de -> en

layer 6, head 8
- (training) ppl : 7.34634, acu : 84.046 %
- (validation) ppl : 8.92932, acu: 66.406 %
- (test) bleu : 0.36006

en -> de

layer 6, head 8
- (training) ppl : 3.94923, acu : 98.799 %
- (validation) ppl : 7.39920, acu: 68.109 %
- (test) bleu : 0.26130


# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert, @Zessay, @JulesGM, @ZiJianZhao, and @huanghoujing.
