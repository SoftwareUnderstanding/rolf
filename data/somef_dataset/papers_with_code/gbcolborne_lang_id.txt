# lang_id

Cuneiform language identification using [BERT](https://arxiv.org/abs/1810.04805).

This code was developed for the shared task on cuneiform language
identification task which was part of the [VarDial 2019 Evaluation
Campaign](https://sites.google.com/view/vardial2019/campaign), and is
largely based on code available in the [Big and Extending Repository
of
Transformers](https://github.com/huggingface/pytorch-pretrained-BERT).


## Requirements

* Python 3 (tested using version 3.7.1)
* PyTorch (tested using version 1.0.0)
* [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples) (tested using version 0.5.1)
* Anything missing? Please let me know.


## Usage 

### Preparing the data

1. Get training and development data. It is assumed to be in a
tab-separated text file containing 2 columns, with text in the first
column and label in the second. The following instructions assume the
labeled training and dev sets (`train.txt` and `dev.txt`) are in a
subdirectory called `data_raw`.

2. Deduplicate training data.

```bash
python map_texts_to_best_labels.py data_raw/train.txt data_raw/train_text_to_best_label.txt
python dedup_labeled_training_file.py data_raw/train.txt data_raw/train_text_to_best_label.txt data_raw/train_dedup.txt
```

3. Gather all labeled data. 

```bash
mkdir data_labeled
cp data_raw/train_dedup.txt data_labeled/train.txt
cp data_raw/dev.txt data_labeled/dev.txt
```

4. Copy dev set as test set. IMPORTANT: this is for illustrative
purposes only, given that we don't have the gold labels of the CLI
test set yet.

```bash
cp data_labeled/dev.txt data_labeled/test.txt
```

5. Strip labels from training set for pretraining. Use `--split`
option to group examples by class, for the sentence pair
classification task.

```bash
mkdir data_unlabeled
python remove_labels.py --split data_labeled/train.txt data_unlabeled/train.txt
```



### Training and testing model

1. Review settings in configuration file `bert_config.json`.

2. Pretrain model on unlabeled training set.

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_BERT_on_2_tasks.py --bert_model_or_config_file bert_config.json --train_file data_unlabeled/train.txt --output_dir model_pretrained --max_seq_length 128 --do_train --train_batch_size 48 --learning_rate 1e-4 --warmup_proportion 0.02 --num_train_epochs 278 --num_gpus 1
```

3. Fine-tune model on labeled training set.

```bash
CUDA_VISIBLE_DEVICES=0 python run_BERT_classifier.py --data_dir data_labeled --bert_model_or_config_file model_pretrained --output_dir model_finetuned --do_train --do_eval --train_batch_size 32 --eval_batch_size 48 --learning_rate 1e-5 --num_train_epochs 8 --num_gpus 1
```


4. Get model's predictions on unlabeled test set.

```bash
python run_BERT_classifier.py --data_dir data_unlabeled --bert_model_or_config_file model_finetuned --output_dir model_predictions --do_predict --eval_batch_size 48  
```


5. Evaluate predictions on test set.

```bash
python evaluate.py data_labeled/test.txt model_predictions/test_pred.txt
```




