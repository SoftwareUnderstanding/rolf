# QA-KD-AL

Improving Question Answering Performance Using Knowledge Distillation and Active Learning

## Requirements

- Python 3.8.3
- PyTorch 1.6.0
- Spacy 2.3.2
- NumPy 1.19.5
- Transformers 4.6.1

## Supported Models

- QANet (Student)
    - QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension [[arXiv: 1804.09541v1](https://arxiv.org/abs/1804.09541v1)]
    - The model implementation is based on [BangLiu/QANet-PyTorch](https://github.com/BangLiu/QANet-PyTorch) and [andy840314/QANet-pytorch-](https://github.com/andy840314/QANet-pytorch-).
- BERT (Teacher)
    - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[arXiv: 1810.04805](https://arxiv.org/abs/1810.04805)]
    - [HuggingFace Transformers](https://github.com/huggingface/transformers) is used for the model implementation.

## Datasets

Use `download.sh` to download and extract the required datasets automatically.

- [GloVe](https://nlp.stanford.edu/projects/glove/)
    - [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)
    - [glove.840B.300d-char.txt](https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt)
- [SQuAD v1.1](rajpurkar.github.io/SQuAD-explorer)
    - [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
    - [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [Adversarial SQuAD](https://worksheets.codalab.org/worksheets/0xc86d3ebe69a3427d91f9aaa63f7d1e7d/)
    - [sample1k-HCVerifyAll](https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/) (AddSent)
    - [sample1k-HCVerifySample](https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/) (AddOneSent)

## Train the Student Model Using Knowledge Distillation

Any BERT-based model selected from [these models](https://huggingface.co/models) can be used as a teacher.

```shell
$ python main.py \
    --train true \
    --epochs 30 \
    --use_cuda true \
    --use_kd true \
    --student "qanet" \
    --batch_size 14 \
    --teacher "bert" \
    --teacher_model_or_path "bert-large-uncased-whole-word-masking-finetuned-squad" \
    --teacher_tokenizer_or_path "bert-large-uncased-whole-word-masking-finetuned-squad" \
    --teacher_batch_size 32 \
    --temperature 10 \
    --alpha 0.7 \
    --interpolation "linear"
```

## Train the Student Model Using Active Learning

The active learning datasets based on the least confidence strategy are provided in `./data/active`.

```shell
$ python main.py \
    --train true \
    --epochs 30 \
    --use_cuda true \
    --use_kd false \
    --student "qanet" \
    --batch_size 14 \
    --train_file ./data/active/train_active_lc5_40.json
```

## Train the Student Model Using Knowledge Distillation and Active Learning 

Before combining knowledge distillation and active learning to train the student model, you have to finetune the teacher model (e.g., BERT-Large) with one of the active learning datasets provided in the `./data/active` directory.

```shell
$ python main.py \
    --train true \
    --epochs 30 \
    --use_cuda true \
    --use_kd false \
    --student "qanet" \
    --batch_size 14 \
    --teacher "bert" \
    --teacher_batch_size 32 \
    --teacher_model_or_path ./processed/bert-finetuned-active-lc5-40 \
    --teacher_tokenizer_or_path ./processed/bert-finetuned-active-lc5-40 \
    --temperature 10 \
    --alpha 0.7 \
    --interpolation "linear" \
    --train_file ./data/active/train_active_lc5_40.json
```

## Evaluate the Student Model

After a successful evaluation, the results will be saved in the `./processed/evaluation` directory by default.

```shell
$ python main.py \
    --evaluate true \
    --use_cuda true \
    --student "qanet" \
    --dev_file ./data/squad/dev-v1.1.json \
    --processed_data_dir ./processed/data \
    --resume ./processed/checkpoints/model_best.pth.tar
```