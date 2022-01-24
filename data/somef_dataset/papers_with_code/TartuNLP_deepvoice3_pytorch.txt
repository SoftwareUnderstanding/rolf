![alt text](assets/banner.jpg)

# Deepvoice3_pytorch

PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

## Deep Voice 3 adaptation for Estonian

This is a modified version of [Ryuichi Yamamoto's implementation](https://github.com/r9y9/deepvoice3_pytorch) of Deep Voice 3 to support Estonian text-to-speech. A Flask API implementation of this code is available [here](https://koodivaramu.eesti.ee/tartunlp/text-to-speech ) and the TTS can be tested with our [web demo](https://www.neurokone.ee).
 
The code contains a submodule for [Estonian TTS preprocessing](https://github.com/TartuNLP/tts_preprocess_et), therefore cloning with the `--recurse-submodules` flag is recommended.

## Pretrained models
Pretrained public model files are available in the [releases section](https://github.com/TartuNLP/deepvoice3_pytorch/releases). It is recommended using the same version of code, as other versions may not be compatible.

## Requirements:

- Python >= 3.5
- CUDA >= 8.0
- PyTorch >= v1.0.0
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.11
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)
- EstNLTK (>= 1.6.0) (Estonian only)

## Installation

Please install packages listed above first, and then

```
git clone https://github.com/TartuNLP/deepvoice3_pytorch --recurse-submodules && cd deepvoice3_pytorch
pip install -e ".[bin]"
```

## Preset parameters

There are many hyper parameters to be turned depends on what model and data you are working on. For typical datasets and models, parameters that known to work good (**preset**) are provided in the repository. See `presets` directory for details. Notice that

1. `preprocess.py`
2. `train.py`
3. `synthesis.py`

accepts `--preset=<json>` optional parameter, which specifies where to load preset parameters. If you are going to use preset parameters, then you must use same `--preset=<json>` throughout preprocessing, training and evaluation. The default preset file for Estonian experiments is `presets/eesti_konekorpus.json`.

## Training
 
To train a multispeaker Estonian TTS model:

```
python preprocess.py eesti_konekorpus $data .data/eesti_konekorpus --speakers Mari,Kalev,Albert,Vesta,Külli,Meelis --preset=presets/eesti_konekorpus.json
python train.py --preset=presets/eesti_konekorpus.json --data-root=./data/eesti_konekorpus --checkpoint-dir=checkpoints/$modelname --log-event-path=log/$modelname
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 10000 steps by default.

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 5. Synthesize from a checkpoint

Given a list of text, `synthesis.py` synthesize audio signals from trained model. Usage is:

```
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir} --preset=presets/eesti_konekorpus.json
```

The text list file should contain one sentence per line.

### Speaker adaptation

If you have very limited data, then you can consider to try fine-turn pre-trained model. For example, using pre-trained model on LJSpeech, you can adapt it to data from VCTK speaker `p225` (30 mins) by the following command:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk_adaptation \
    --preset=presets/deepvoice3_ljspeech.json \
    --log-event-path=log/deepvoice3_vctk_adaptation \
    --restore-parts="20171213_deepvoice3_checkpoint_step000210000.pth"
    --speaker-id=0
```

From my experience, it can get reasonable speech quality very quickly rather than training the model from scratch.

There are two important options used above:

- `--restore-parts=<N>`: It specifies where to load model parameters. The differences from the option `--checkpoint=<N>` are 1) `--restore-parts=<N>` ignores all invalid parameters, while `--checkpoint=<N>` doesn't. 2) `--restore-parts=<N>` tell trainer to start from 0-step, while `--checkpoint=<N>` tell trainer to continue from last step. `--checkpoint=<N>` should be ok if you are using exactly same model and continue to train, but it would be useful if you want to customize your model architecture and take advantages of pre-trained model.
- `--speaker-id=<N>`: It specifies what speaker of data is used for training. This should only be specified if you are using multi-speaker dataset. As for VCTK, speaker id is automatically assigned incrementally (0, 1, ..., 107) according to the `speaker_info.txt` in the dataset.

If you are training multi-speaker model, speaker adaptation will only work **when `n_speakers` is identical**.

## Troubleshooting

### [#5](https://github.com/r9y9/deepvoice3_pytorch/issues/5) RuntimeError: main thread is not in main loop


This may happen depending on backends you have for matplotlib. Try changing backend for matplotlib and see if it works as follows:

```
MPLBACKEND=Qt5Agg python train.py ${args...}
```

In [#78](https://github.com/r9y9/deepvoice3_pytorch/pull/78#issuecomment-385327057), engiecat reported that changing the backend of matplotlib from Tkinter(TkAgg) to PyQt5(Qt5Agg) fixed the problem.

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/keithito/tacotron
- https://github.com/facebookresearch/fairseq-py

Banner and logo created by [@jraulhernandezi](https://github.com/jraulhernandezi) ([#76](https://github.com/r9y9/deepvoice3_pytorch/issues/76))
