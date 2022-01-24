# Text to python code

## Task Definition

The task is to generate code from natural language, and evaluted by bleu (https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Dataset

The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf) and we filter the dataset as the following:

- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

### Download data and preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```


### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

  - **repo:** the owner/repo

  - **path:** the full path to the original file

  - **func_name:** the function or method name

  - **original_string:** the raw string before tokenization or parsing

  - **language:** the programming language

  - **code/function:** the part of the `original_string` that is code

  - **code_tokens/function_tokens:** tokenized version of `code`

  - **docstring:** the top-level comment or docstring, if it exists in the original string

  - **docstring_tokens:** tokenized version of `docstring`


## Evaluator

We provide a script to evaluate predictions for this task, and report smoothed bleu score.


## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. The encoder is CodeBERT and the decoder is 6-layers Transformer.

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
cd code
lang=ruby #programming language
lr=5e-5
batch_size=8
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
```


### Inference

```shell
batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

### Evaluation

```shell
python ../evaluator/evaluator.py model/$lang/test_1.gold < model/$lang/test_1.output
```

## Result

The results on the test set are shown as below:

| Model        |     2K    |     10K   |    30K    | 
| -----------  | :-------: | :-------: | :-------: |
| Transformers |   4.58    |   6.89    |    7,19   |
| Our model    | **10.24** | **12.24** | **13.04** |


## Reference
<pre><code>@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}</code></pre>
