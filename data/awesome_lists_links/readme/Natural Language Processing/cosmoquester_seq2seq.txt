# seq2seq

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/seq2seq.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/seq2seq)
[![codecov](https://codecov.io/gh/cosmoquester/seq2seq/branch/master/graph/badge.svg?token=qZjiAuSYiw)](https://codecov.io/gh/cosmoquester/seq2seq)

This is seq2seq model structures with Tensorflow 2.

There are three model architectures, RNNSeq2Seq, RNNSeq2SeqWithAttention, TransformerSeq2Seq.

This repository contains train, evaulate, inference, converting to savedmodel format scripts.

이 코드를 이용해 학습하고 실험한 결과는 [Tensorflow2 기반 Seq2Seq 모델, 학습, 서빙 코드 구현](https://cosmoquester.github.io/seq2seq/) 에서 볼 수 있습니다. (한국어)
# Train

## Example

You can start training by running script like below
```sh
$ python -m scripts.train \
	--dataset-path "data/*.txt" \
	--batch-size 2048 --dev-batch-size 2048 \
	--epoch 90 --steps-per-epoch 250 --auto-encoding \
	--learning-rate 2e-4 \
	--device gpu \
	--tensorboard-update-freq 50 --model-name TransformerSeq2Seq --model-config-path resources/configs/transformer.yml
```

## Arguments

```
File Paths:
  --model-name MODEL_NAME
                        Seq2seq model name
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --dataset-path DATASET_PATH
                        a text file or multiple files ex) *.txt
  --pretrained-model-path PRETRAINED_MODEL_PATH
                        pretrained model checkpoint
  --output-path OUTPUT_PATH
                        output directory to save log and model checkpoints
  --sp-model-path SP_MODEL_PATH

Training Parameters:
  --epochs EPOCHS
  --steps-per-epoch STEPS_PER_EPOCH
  --learning-rate LEARNING_RATE
  --min-learning-rate MIN_LEARNING_RATE
  --warm-up-rate WARM_UP_RATE
  --batch-size BATCH_SIZE
  --dev-batch-size DEV_BATCH_SIZE
  --num-dev-dataset NUM_DEV_DATASET
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
  --prefetch-buffer-size PREFETCH_BUFFER_SIZE
  --max-sequence-length MAX_SEQUENCE_LENGTH

Other settings:
  --tensorboard-update-freq TENSORBOARD_UPDATE_FREQ
                        log losses and metrics every after this value step
  --disable-mixed-precision
                        Use mixed precision FP16
  --auto-encoding       train by auto encoding with text lines dataset
  --use-tfrecord        train using tfrecord dataset
  --debug-nan-loss      Trainin with this flag, print the number of Nan loss
                        (not supported on TPU)
  --device {CPU,GPU,TPU}
                        device to train model
  --max-over-sequence-policy {filter,slice}
                        Policy for sequences of which length is over the max
```
- `model-name` is seq2seq model class name, so one of (RNNSeq2Seq, RNNSeq2SeqWithAttention, TransformerSeq2Seq)
- `model-config-path` is model config file path. model config file describe model parameter. There are default model configs in `resources/configs`
- `dataset-path` is dataset file glob expression. dataset file format is tsv file without header having two columns sequence A, sequenceB
    model training to predict sequence B when we inputs sequence A. However, when we use `auto-encoding` option, dataset format is just lines of text.
    So, model will be trained to echo texts.
- `sp-model-path` is sentencepiece model path to tokenize text.
- `disable-mixed-precision` is to disable fp16 mixed precision. Mixed precision is on as default.
- `device` is training device, one of (cpu, gpu, tpu) But tpu is not supported yet. If you want to train on TPU, `use-tfrecord` option is necessary.
    TFRecord can be made with "scripts/make_tfrecord.py" python script.
When ending of training, the model checkpoints and tensorboard log files are saved to output directory.

# Evaluate

## Example

You can start training by running script like below
```sh
$ python -m scripts.evaluate \
    --model-path ~/Downloads/output/models/model-50epoch-nanloss_0.3870acc.ckpt \
    --model-config-path ~/Downloads/output/model_config.yml  \
    --dataset-path test.txt \
    --auto-encoding \
    --beam-size 2 \
    --disable-mixed-precision

[skip some messy logs...]
[2020-12-20 01:42:28,308] RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.
DEBUG:tensorflow:RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.
[2020-12-20 01:42:28,311] RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.
DEBUG:tensorflow:RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.
[2020-12-20 01:42:28,315] RNN `implementation=2` is not supported when `recurrent_dropout` is set. Using `implementation=1`.
[2020-12-20 01:42:30,963] Loaded weights of model
Perplexity: 17.618855794270832, BLEU: 0.07615733809469007: : 1it [00:04,  4.38s/it]
[2020-12-20 01:42:35,347] Finished evalaution!
[2020-12-20 01:42:35,348] Perplexity: 17.618855794270832, BLEU: 0.07615733809469007
```
Results is ppl and BLEU.

## Arguments

```sh
File Paths:
  --model-name MODEL_NAME
                        Seq2seq model name
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --dataset-path DATASET_PATH
                        a tsv file or multiple files ex) *.tsv
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --sp-model-path SP_MODEL_PATH

Inference Parameters:
  --batch-size BATCH_SIZE
  --prefetch-buffer-size PREFETCH_BUFFER_SIZE
  --max-sequence-length MAX_SEQUENCE_LENGTH
  --header              use this flag if dataset (tsv file) has header
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size

Other settings:
  --disable-mixed-precision
                        Use mixed precision FP16
  --auto-encoding       evaluate by autoencoding performance dataset format is
                        lines of texts (.txt)
  --device DEVICE       device to train model
```
- Most of arugments is same as training script.
- `beam-size` is beam search parameter. When this is less than two or not given, use greedy search.

# Inference

## Example

You can start training by running script like below
```sh
$ python -m scripts.inference \
    --dataset-path test.txt \
    --model-path ~/Downloads/output/models/model-50epoch-nanloss_0.3870acc.ckpt \
    --output-path out.txt \
    --save-pair

[skip some messy logs...]
[2020-12-20 01:52:27,856] Loaded weights of model
[2020-12-20 01:52:27,857] Start Inference
[2020-12-20 01:52:35,629] Ended Inference, Start to save...
[2020-12-20 01:52:35,631] Saved (original sentence,decoded sentence) pairs to out.txt
```

## Arguments

```
File Paths:
  --model-name MODEL_NAME
                        Seq2seq model name
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --dataset-path DATASET_PATH
                        a text file or multiple files ex) *.txt
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --output-path OUTPUT_PATH
                        output file path to save generated sentences
  --sp-model-path SP_MODEL_PATH

Inference Parameters:
  --batch-size BATCH_SIZE
  --prefetch-buffer-size PREFETCH_BUFFER_SIZE
  --max-sequence-length MAX_SEQUENCE_LENGTH
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size

Other settings:
  --disable-mixed-precision
                        Use mixed precision FP16
  --save-pair           save result as the pairs of original and decoded
                        sentences
  --device DEVICE       device to train model
```
- When use `save-pair` option, save with original sentence and generated sentence with tsv format. If not, save only generated sentences.

# Interactive

## Example

You can test your trained model interactively. If you want to finish, just enter to put empty input.
```sh
$ python -m scripts.interactive \
    --model-name TransformerSeq2Seq \
    --model-path ~/model-28epoch-0.1396loss_0.9812acc.ckpt \
    --model-config-path resources/configs/transformer.yml

[2021-04-30 00:27:00,037] Loaded weights of model
Please Input Text: 너 이름이 뭐야?
2021-04-30 00:27:23.437004: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-04-30 00:27:23.705647: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
Output: 너 이름이 뭐야?, Perplexity: 1.0628
Please Input Text: 근데 어쩌라는 걸까
Output: 근데 어쩌라는 걸까, Perplexity: 1.0594
Please Input Text:  헤헤헤헤
Output: 헤헤헤헤, Perplexity: 1.4151
Please Input Text:
```

## Arguments

```
File Paths:
  --model-name MODEL_NAME
                        Seq2seq model name
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --model-path MODEL_PATH
                        pretrained model checkpoint
  --sp-model-path SP_MODEL_PATH

Inference Parameters:
  --batch-size BATCH_SIZE
  --prefetch-buffer-size PREFETCH_BUFFER_SIZE
  --max-sequence-length MAX_SEQUENCE_LENGTH
  --pad-id PAD_ID       Pad token id when tokenize with sentencepiece
  --beam-size BEAM_SIZE
                        not given, use greedy search else beam search with
                        this value as beam size

Other settings:
  --mixed-precision     Use mixed precision FP16
  --device DEVICE       device to train model
```

# Convert to savedmodel

## Example

You can simply convert model checkpoint to savedmodel format.
```sh
$ python -m scripts.convert_to_savedmodel \
    --model-name RNNSeq2SeqWithAttention \
    --model-config-path ~/Downloads/output/model_config.yml \
    --model-weight-path ~/Downloads/output/models/model-50epoch-nanloss_0.3870acc.ckpt \
    --output-path seq2seq-model/1

[skip some messy logs...]
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
INFO:tensorflow:Assets written to: seq2seq-model/1/assets
[2020-12-20 01:58:49,424] Assets written to: seq2seq-model/1/assets
[2020-12-20 01:58:51,285] Saved model to seq2seq-model/1
```
If you make savedmodel by using this script, tokenize is included in savedmodel so you can use generate sequence without tokenizer or vocab.

## Arguments

```
Arguments:
  --model-name MODEL_NAME
                        Seq2seq model name
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --model-weight-path MODEL_WEIGHT_PATH
                        Model weight file path saved in training
  --sp-model-path SPM_MODEL_PATH
                        sp tokenizer model path
  --output-path OUTPUT_PATH
                        Savedmodel path

Search Method Configs:
  --pad-id PAD_ID       Pad token id when tokenize with sentencepiece
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        Max number of tokens including bos, eos
  --alpha ALPHA         length penalty control variable when beam searching
  --beta BETA           length penalty control variable when beam searching
```

## Use of savedmodel

```sh
$ docker run -v `pwd`/seq2seq-model:/models/seq2seq -e MODEL_NAME=seq2seq -p 8501:8501 -dt tensorflow/serving
```
You can open tensorflow serving server.

```sh
$ curl -XPOST localhost:8501/v1/models/seq2seq:predict -d '{"inputs":["안녕하세요", "나는 오늘 밥을 먹었다", "아니 지금 뭐라고요?, 그게 대체 무슨 말이에요!!"]}'
{
    "outputs": {
        "perplexity": [
            1.00468457,
            1.06678605,
            1.04327798
        ],
        "sentences": [
            "안녕하세요",
            "나는 오늘 밥을 먹었다",
            "아니 지금 뭐라고요?, 그게 대체 무슨 말이에요!!"
        ]
    }
}
```
- By default, signature function is greedy search. Like above example, you can send texts then receice ppl and gernerated texts.

```sh
$ curl -XPOST localhost:8501/v1/models/seq2seq:predict -d '{"inputs":{"texts":["반갑습니다", "학교가기 싫다"], "beam_size":3}, "signature_name":"beam_search"}'
{
    "outputs": {
        "sentences": [
            [
                "반갑습니다",
                "반갑습니다",
                "반갑습니다"
            ],
            [
                "학교가기 싫다",
                "학교가기놔",
                "학교가기 챙겨"
            ]
        ],
        "perplexity": [
            [
                1.0299294,
                1.0299294,
                1.0299294
            ],
            [
                1.22807097,
                1.54527545,
                1.56684875
            ]
        ]
    }
}
```
- If you want to inference by beam searching, set `signature_name` as beam_search and request with beam_size.
- Response also contains beam size number of texts per an example.

# References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
