# DJL
AI DJ, Drop The Beat.

## Required Knowledge
### Beats
  - **BPM**: Beats Per Minutes
  - **QPM**: Quarter-notes Per Minutes
  - **Difference**: They are the same with time signatures such as 3/4 and 4/4, which have beats of a quarter-note in length. However, time signatures based on eighth-notes have a beat length only half the size, which should normally play twice as fast.

### MIDI (Musical Instrument Digital Interface)
- A unified language that allows different musical pieces of equipment to communicate with one another 
- It is not an audio signal, it is a trigger for communicating to another device, it transmits information such as tonation, the velocity of the performance, volume, etc.

### Musical Scores
- a written form of a musical composition.
- parts for different instruments appear on separate staves on large pages.

## Network To Try
### [MusicVAE](https://github.com/tensorflow/magenta/tree/master/magenta/models/music_vae)
A hierarchical recurrent variational autoencoder for music.

MusicVAE learns a latent space of musical sequences, providing different modes of interactive musical creation, including:
- random sampling from the prior distribution
- interpolation between existing sequences
- manipulation of existing sequences via attribute vectors or a latent constraint model.

### [Melody RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn#melody-rnn)
This configuration acts as a baseline for melody generation with an LSTM model. It uses basic one-hot encoding to represent extracted melodies as input to the LSTM. For training, all examples are transposed to the MIDI pitch range [48, 84] and outputs will also be in this range.

### [MelNet](https://sjvasquez.github.io/blog/melnet/)
Able to generate high-fidelity audio samples that capture structure at timescales that time-domain models have yet to achieve. Powered by Recurrent Convolutional Neural Network.
- Paper: https://arxiv.org/pdf/1906.01083.pdf
- Instead of modeling a 1-D time-domain wave form, we can model a 2-D time-frequency representation: Spectrogram.
  - **Spectrogram**: a picture of sound. The x-axis is Time, the y-axis is Frequency, and the brightness (lightness or darkness) at any given point represents the energy at that point.

### [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
The result isn't ideal, the sound is still too robotic.
- Paper: https://arxiv.org/abs/1609.03499
