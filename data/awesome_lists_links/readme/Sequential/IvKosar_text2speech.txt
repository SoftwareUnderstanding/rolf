# Sequence translation: text-to-speech

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

### Content

- [Stages of project](#stages-of-project)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [CBHG submodule](#cbhg-submodule)
- [Dependencies](#dependencies)
- [Literature and references](#literature-and-references)


## Stages of project:
- [x] Find dataset
- [x] Data pre-processing
    - [x] Text pre-processing
    - [x] Audio pre-processing
- [x] Sequence model architecture
    - [x] Choose state of the art architecture - **Tacotron**
    - [x] Create architecture of the model from the paper
- [x] Training
    - [x] Implement train module
    - [x] Add tensorboard
    - [x] Train model
- [x] Evaluation
    - [x] Inference module


## Usage

#### Training model
```bash
$ python train.py --config configs/config.yaml  
```

We have trained the model for 45k iterations.

#### Inference
To use the model we created jupyter notebook (_inference.ipynb_).

Here is the example of usage:

```python
text0 = "It was a great day"
wav, alignment, spectrogram = inference(text0)
IPython.display.display(Audio(wav, rate=audio_configs["sample_rate"]))
visualize_spectrogram(spectrogram, alignment)
```

The output is generated audio. It has taken ~2 seconds for generation on GPU GeForce GTX 1060. You can listen to it in the [ipynb](inference.ipynb) notebook or [here](http://marianpetruk.github.com/projects/text2speech/generated/itwaagrda.wav).

![example](imgs/ex1.png)

##### Other generated examples

| Text| `It was a great day` | `I love Machine Learning` | `My name is Pytorch and I live on cuda` | `I gonna take my horse to the old town road` |
|-------|-------|-------|-------|-------|
| Generated speech in wav | [link](http://marianpetruk.github.com/projects/text2speech/generated/itwaagrda.wav) , [alt_link](generated_audio/wav/itwaagrda.wav) | [link](http://marianpetruk.github.com/projects/text2speech/generated/ilomale.wav) , [alt_link](generated_audio/wav/ilomale.wav) | [link](http://marianpetruk.github.com/projects/text2speech/generated/mynaispyanilioncu.wav) , [alt_link](generated_audio/wav/mynaispyanilioncu.wav) | [link](http://marianpetruk.github.com/projects/text2speech/generated/igotamyhototholtoro.wav) , [alt_link](generated_audio/wav/igotamyhototholtoro.wav) |
| Converted wav to mp3 | [link](generated_audio/converted_to_mp3/itwaagrda.mp3) | [link](generated_audio/converted_to_mp3/ilomale.mp3) | [link](generated_audio/converted_to_mp3/mynaispyanilioncu.mp3) | [link](generated_audio/converted_to_mp3/igotamyhototholtoro.mp3) |

## Dataset
For our project we choose to use [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).
It consists of 13100 audio clips of a single speaker with transcriptions to every clip.
Total number of words is 225715 and total length of all audio is almost 24 hours.

## Model Architecture
![tacotron architecture diagram](imgs/Tacotron.jpg)

The model takes characters as input and outputs the corresponding raw spectrogram, which is then fed to the Griffin-Lim reconstruction algorithm to synthesize speech.

### CBHG submodule
![CBHG module](imgs/CBHG.png)

CBHG  consists  of  a bank of 1-D convolutional filters,  followed by highway networks and bidirectional gated recurrent unit (GRU)  recurrent neural net (RNN). 
__CBHG is a powerful module for extracting representations from sequences.__


#### Model weights
You can find model weights @ [link](https://drive.google.com/file/d/1ioRZOR1vD-qPpDIoA9Mwi4hZ7w3HFGuW/view?usp=sharing).

## Dependencies

You can install all required dependencies with: 
```bash
$ pip install -r requirements.txt
```

You can also install the latest packages manually with:

  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/numpy/badges/version.svg)](https://anaconda.org/anaconda/numpy): `conda install numpy`
  - [![Anaconda-Server Badge](https://anaconda.org/anaconda/scipy/badges/version.svg)](https://anaconda.org/anaconda/scipy): `conda install scipy`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/tqdm/badges/version.svg)](https://anaconda.org/conda-forge/tqdm): `conda install -c conda-forge tqdm`
  - [![PyPI version](https://badge.fury.io/py/tensorboardX.svg)](https://badge.fury.io/py/tensorboardX): `pip install tensorboardX`
  - [![Anaconda-Server Badge](https://anaconda.org/pytorch/pytorch/badges/version.svg)](https://anaconda.org/pytorch/pytorch):  `conda install -c pytorch pytorch`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/matplotlib/badges/version.svg)](https://anaconda.org/conda-forge/matplotlib): `conda install -c conda-forge matplotlib`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pandas/badges/version.svg)](https://anaconda.org/conda-forge/pandas): ` conda install -c conda-forge pandas`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/librosa/badges/version.svg)](https://anaconda.org/conda-forge/librosa): `conda install -c conda-forge librosa`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/unidecode/badges/version.svg)](https://anaconda.org/conda-forge/unidecode): `conda install -c conda-forge unidecode`
  - [![Anaconda-Server Badge](https://anaconda.org/conda-forge/yaml/badges/version.svg)](https://anaconda.org/conda-forge/yaml) : `conda install -c conda-forge yaml`
  - [![PyPI version](https://badge.fury.io/py/SoundFile.svg)](https://badge.fury.io/py/SoundFile): `pip install SoundFile`

## Literature and references:
- Tacotron: Towards End-to-End Speech Synthesis	[arXiv:1703.10135](https://arxiv.org/abs/1703.10135) [cs.CL]
- [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html) @ [Dive into Deep Learning](https://d2l.ai/index.html) interactive book
- [Text to Speech Deep Learning Architectures](http://www.erogol.com/text-speech-deep-learning-architectures/)
- [Deep Learning for Audio](http://slazebni.cs.illinois.edu/spring17/lec26_audio.pdf) Y. Fan, M. Potok, C. Shroba
- [Deep Learning for Text-to-Speech Synthesis, using the Merlin toolkit](http://www.speech.zone/courses/one-off/merlin-interspeech2017/)
- [Babble-rnn: Generating speech from speech with LSTM networks](http://babble-rnn.consected.com/docs/babble-rnn-generating-speech-from-speech-post.html)
- https://github.com/r9y9/tacotron_pytorch
- https://github.com/keithito/tacotron
- https://github.com/mozilla/TTS
- https://github.com/Kyubyong/tacotron
- [The Centre for Speech Technology Research](http://www.cstr.ed.ac.uk/)
- [Preparing Data for Training an HTS Voice](http://www.cs.columbia.edu/~ecooper/tts/data.html)
- [awesome speech synthesis/recognition papers](http://rodrigo.ebrmx.com/github_/zzw922cn/awesome-speech-recognition-speech-synthesis-papers)
