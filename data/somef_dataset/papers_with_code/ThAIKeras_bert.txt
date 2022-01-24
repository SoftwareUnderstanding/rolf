
# BERT-th

Google's [**BERT**](https://github.com/google-research/bert) is currently the state-of-the-art method of pre-training text representations which additionally provides multilingual models. ~~Unfortunately, Thai is the only one in 103 languages that is excluded due to difficulties in word segmentation.~~

BERT-th presents the Thai-only pre-trained model based on the BERT-Base structure. It is now available to download.
* **[`BERT-Base, Thai`](https://drive.google.com/open?id=1J3uuXZr_Se_XIFHj7zlTJ-C9wzI9W_ot)**: BERT-Base architecture, Thai-only model

BERT-th also includes relevant codes and scripts along with the pre-trained model, all of which are the modified versions of those in the original BERT project.

## Preprocessing

### Data Source

Training data for BERT-th come from [the latest article dump of Thai Wikipedia](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) on November 2, 2018. The raw texts are extracted by using [WikiExtractor](https://github.com/attardi/wikiextractor).

### Sentence Segmentation

Input data need to be segmented into separate sentences before further processing by BERT modules. Since Thai language has no explicit marker at the end of a sentence, it is quite problematic to pinpoint sentence boundaries. To the best of our knowledge, there is still no implementation of Thai sentence segmentation elsewhere. So, in this project, sentence segmentation is done by applying simple heuristics, considering spaces, sentence length and common conjunctions.

After preprocessing, the training corpus consists of approximately 2 million sentences and 40 million words (counting words after word segmentation by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)). The plain and segmented texts can be downloaded **[`here`](https://drive.google.com/file/d/1QZSOpikO6Qc02gRmyeb_UiRLtTmUwGz1/view?usp=sharing)**.

## Tokenization

BERT uses [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) as a tokenization mechanism. But it is Google internal, we cannot apply existing Thai word segmentation and then utilize WordPiece to learn the set of subword units. The best alternative is [SentencePiece](https://github.com/google/sentencepiece) which implements [BPE](https://arxiv.org/abs/1508.07909) and needs no word segmentation.

In this project, we adopt a pre-trained Thai SentencePiece model from [BPEmb](https://github.com/bheinzerling/bpemb). The model of 25000 vocabularies is chosen and the vocabulary file has to be augmented with BERT's special characters, including '[PAD]', '[CLS]', '[SEP]' and '[MASK]'. The model and vocabulary files can be downloaded **[`here`](https://drive.google.com/file/d/1F7pCgt3vPlarI9RxKtOZUrC_67KMNQ1W/view?usp=sharing)**.

`SentencePiece` and `bpe_helper.py` from BPEmb are both used to tokenize data. `ThaiTokenizer class` has been added to BERT's `tokenization.py` for tokenizing Thai texts.

## Pre-training

The data can be prepared before pre-training by using this script.

```shell
export BPE_DIR=/path/to/bpe
export TEXT_DIR=/path/to/text
export DATA_DIR=/path/to/data

python create_pretraining_data.py \
  --input_file=$TEXT_DIR/thaiwikitext_sentseg \
  --output_file=$DATA_DIR/tf_examples.tfrecord \
  --vocab_file=$BPE_DIR/th.wiki.bpe.op25000.vocab \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --thai_text=True \
  --spm_file=$BPE_DIR/th.wiki.bpe.op25000.model
```

Then, the following script can be run to learn a model from scratch.

```shell
export DATA_DIR=/path/to/data
export BERT_BASE_DIR=/path/to/bert_base

python run_pretraining.py \
  --input_file=$DATA_DIR/tf_examples.tfrecord \
  --output_dir=$BERT_BASE_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=100000 \
  --learning_rate=1e-4 \
  --save_checkpoints_steps=200000
```

We have trained the model for 1 million steps. On Tesla K80 GPU, it took around 20 days to complete. Though, we provide a snapshot at 0.8 million steps because it yields better results for downstream classification tasks.

## Downstream Classification Tasks

### XNLI

[XNLI](http://www.nyu.edu/projects/bowman/xnli/) is a dataset for evaluating a cross-lingual inferential classification task. The development and test sets contain 15 languages which data are thoroughly edited. The machine-translated versions of training data are also provided.

The Thai-only pre-trained BERT model can be applied to the XNLI task by using training data which are translated to Thai. Spaces between words in the training data need to be removed to make them consistent with inputs in the pre-training step. The processed files of XNLI related to Thai language can be downloaded **[`here`](https://drive.google.com/file/d/1ZAk1JfR6a0TSCkeyQ-EkRtk1w_mQDWFG/view?usp=sharing)**.

Afterwards, the XNLI task can be learned by using this script.

```shell
export BPE_DIR=/path/to/bpe
export XNLI_DIR=/path/to/xnli
export OUTPUT_DIR=/path/to/output
export BERT_BASE_DIR=/path/to/bert_base

python run_classifier.py \
  --task_name=XNLI \
  --do_train=true \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BPE_DIR/th.wiki.bpe.op25000.vocab \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$OUTPUT_DIR \
  --xnli_language=th \
  --spm_file=$BPE_DIR/th.wiki.bpe.op25000.model
```

This table compares the Thai-only model with XNLI baselines and the Multilingual Cased model which is also trained by using translated data.

<!-- Use html table because github markdown doesn't support colspan -->
<table>
  <tr>
    <td colspan="2" align="center"><b>XNLI Baseline</b></td>
    <td colspan="2" align="center"><b>BERT</b></td>
  </tr>
  <tr>
    <td align="center">Translate Train</td>
    <td align="center">Translate Test</td>
    <td align="center">Multilingual Model</td>
    <td align="center">Thai-only Model</td>
  </tr>
    <td align="center">62.8</td>
    <td align="center">64.4</td>
    <td align="center">66.1</td>
    <td align="center"><b>68.9</b></td>
</table>

### Wongnai Review Dataset

Wongnai Review Dataset collects restaurant reviews and ratings from [Wongnai](https://www.wongnai.com/) website. The task is to classify a review into one of five ratings (1 to 5 stars). The dataset can be downloaded **[`here`](https://github.com/wongnai/wongnai-corpus)** and the following script can be run to use the Thai-only model for this task.

```shell
export BPE_DIR=/path/to/bpe
export WONGNAI_DIR=/path/to/wongnai
export OUTPUT_DIR=/path/to/output
export BERT_BASE_DIR=/path/to/bert_base

python run_classifier.py \
  --task_name=wongnai \
  --do_train=true \
  --do_predict=true \
  --data_dir=$WONGNAI_DIR \
  --vocab_file=$BPE_DIR/th.wiki.bpe.op25000.vocab \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$OUTPUT_DIR \
  --spm_file=$BPE_DIR/th.wiki.bpe.op25000.model
```

Without additional preprocessing and further fine-tuning, the Thai-only BERT model can achieve 0.56612 and 0.57057 for public and private test-set scores respectively.
