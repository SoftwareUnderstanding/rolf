
# TorchRosa TTS

## What is this?

This project contains files I put together to try out creating a text-to-speech
system from scratch. The system uses a PyTorch model to learn a VAE that is
used to learn an embedding that plugs into the decoder. The result is something
that vaguely resembles my voice if it was sent through a blender.

## Installation

I will try to get around to writing a requirements.txt, but I will summarize the
requirements as best as possible. For this project, I installed:

```
python3
pytorch
librosa
pyaudio
g2p_en
scipy
numpy
```

## Usage

I will provide the PyTorch model I generated for my own voice for free. To use
this model, the interface is provided through generate.py. I designed it so
additional models could be setup for the class if desired. The simplest way to
use it is to run

```
python3 generate.py --text "Hello world"
```

That's it! It should cobble together a speech sample in `output.wav`

### What if I want to use my own voice?

So this is the fun part. So in order to do this, you will need to record
yourself speaking the majority of the phonemes. Some of the phonemes were
filtered out because they didn't provide continuous sounds, and are replaced
by ones that do provide continuous sounds that are placed in sequence.
These can be seen in `util/preprocess.py`. For a full list of phonemes, run

```
python phoneme_handler.py list-phonemes
```

When I recorded my voice, I practiced making the phoneme sounds. To actually
record, I provide a convenient utility for this.

```
python phoneme_handler.py record-phoneme AA
```

This will record the phoneme AA. When you run it, it will prompt you to press
enter when you are ready. I recommend beginning to make the phoneme sound
immediately after the button press, and continuing until the program terminates
for best results. I do not recommend this if you have pre-existing health
issues that impact your breathing or lung capacity. It is possible to reduce
the recording length in the program, as there is a constant where I se the
utterance length to five seconds.

If you have issues with missing directories, creating them should resolve most
issues.

Once the phonemes are recorded, you need to generate the spectrograms. This is
done by running ```python generate_spectrograms.py```. This will create Mel
spectrograms of each audio file recorded and save them as numpy files within
the package.

Finally, you need to train the model. Running `python train.py` will train the
VAE model by default. If you wish to do hyperparameter tuning, you can edit
`vae/hyperparameters.py`. Note that I was lazy with my train-test split setup,
so there will be contamination. But I found that the hyperparameters there
worked okay for my files.

Once `train.py` has been run, it will save `model.pt` in your directory. If
this file already exists, it will be overwritten. Once it exists, you can
use the model.

### What if I want to add my own words?

I have already provided words that cover all of the phonemes. However, it may
be useful to provide your own to help you figure out the pronounciations.
To do this, edit the word list, and then run

```
python generate-metadata.py
```

This will generate the CSV file in the current directory.

## Future Ideas

In the future, I might want to mess with using GAN to improve the results.
I have a couple of ideas here:

- Improving the quality of the phoneme sound bites by training a discriminator to learn realness of phoneme sounds
- Learning a model that converts the generated spectrogram to a more realistic sounding audio

I do also wonder if there is a way to speed up the WAV generation process, as
it is quite slow on a laptop. One investigation that could be done would be to
see if FFT can be used to generate better results.

## References

VAE: https://arxiv.org/abs/1312.6114

Inspiration for Mel spectrogram use: https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html

