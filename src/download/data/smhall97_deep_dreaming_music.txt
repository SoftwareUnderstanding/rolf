# Deep Dreaming Music
***[Neuromatch Academy Deep Learning 2021 project](https://docs.google.com/presentation/d/1ccBBTnq6kJ8Kvi7bRW69x3lTiZRxoFsAwzD56nLdwHQ/present?slide=id.ge51dc41796_2_83) by [Maryam Faramarzi](https://github.com/MaryamFaramarzi), [Siobhan Hall](https://github.com/smhall97), [Máté Mohácsi](https://github.com/mohacsimate), [Pablo Oyarzo](https://github.com/oyarzou), [Jonathan Reus](https://github.com/jreus) and [Katherine Baquero](https://github.com/KatherineBaq)***

*Special credit to our TAs: *[Pedro F da Costa](https://github.com/PedroFerreiradaCosta)* and *[Beatrix BenkÅ](https://github.com/bbeatrix)**

---

## Background
During the last decade CNNs have become increasingly powerful as models for computer vision. Their development has been carried conjointly with the exploration of their internal informational structure (1) and feature-inference processes. One of these approaches known as “dreaming”, initially developed by Google (2), has proven to be an effective method to maximize features and to take convolutional neural networks to the territory of stimuli generation. 
In this work, we ask how audible waveforms can be reconstructed from features by exploring dreaming algorithms as a method. This method has the capacity to work as an introspective technique for understanding internal network representations, as well as potentially acting as a generative approach for novel audio output. 
CNNs have been used to achieve state-of-the-art performance classifying music genres from (Mel Scale) spectrograms (3), thus we choose to explore the problem of music genre classification as a starting point for exploring internal representations. While most literature on CNN-based music genre classifiers use Mel-scale Spectrograms, this audio representation can be potentially limiting in the specific situation of deep dreaming on spectrograms, where the end goal is to reconstruct an audible time-domain waveform. It is for this reason we investigated the following audio transforms: Short-Time-Fourier Transforms (STFTs) and Mel-spectrograms. These will be collectively referred to as Spectrograms throughout.


## Aims
- To determine if audible waveforms can be reconstructed using the "dreaming" process from features learned by a convolutional neural network. 
- To investigate the best approach to training a classifier as well as choosing the type of audio data transform that can be reconverted to music once “dreamed” upon. 

---
## Methods
![Overall pipeline - from classification to dreaming](https://github.com/smhall97/deep_dreaming_music/blob/main/Pipelines/Overall%20pipeline.png)

### Dataset
[Exploring the dataset](https://github.com/smhall97/deep_dreaming_music/blob/main/Audio%20data%20preparation/AudioFormatConversion.ipynb)

[Preparing the data](https://github.com/smhall97/deep_dreaming_music/blob/main/Audio%20data%20preparation/wav2spectrograms.ipynb)

**[GTZAN dataset](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)**
- 1000 audio tracks (length: 30 seconds; 22050 Hz Mono 16-bit audio files in .wav format)
- 10 genres, with 100 examples each (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
- The STFTs were computed with the real and imaginary part, as well as only the real component. Both of these versions were used separately 

### Models
**Reference papers**

- [VGG16](https://arxiv.org/abs/1409.1556)
- [Inception_v3](https://arxiv.org/abs/1512.00567)

**Code:**

- [Training convolutional neural networks](https://github.com/smhall97/deep_dreaming_music/blob/main/Combined_CNN_Training.ipynb)
- [Feature visualisation](https://github.com/smhall97/deep_dreaming_music/blob/main/Feature_visualisation_VGG16.ipynb)


---
## Dreaming
[Deep Dreaming Music Code](https://github.com/smhall97/deep_dreaming_music/blob/main/Deep_Dream_Music.ipynb)


![Deep Dream Pipeline](https://github.com/smhall97/deep_dreaming_music/blob/main/Pipelines/Deep%20Dream%20Pipeline.png)

#### Update strategies during dreaming: additive and subtractive
For either approach, one can choose to normalize the gradient or not (which usually leads to faster learning at the cost of sometimes opting for less nuanced dream features)

**Additive** 

![additive](https://github.com/smhall97/deep_dreaming_music/blob/main/results/update_additive.gif)


**Subtractive**

![subtractive](https://github.com/smhall97/deep_dreaming_music/blob/main/results/update_subtractive.gif)


### Optimization Functions
**Maximize activation of a single genre**
![Maximize activation of a single genre](https://github.com/smhall97/deep_dreaming_music/blob/main/results/optimize_activation.png)


**Maximize activation difference between a single genre and another (or all other) genre(s)**
![Maximise differences](https://github.com/smhall97/deep_dreaming_music/blob/main/results/optimize_difference.png)

---
## Findings
### Classification accuracies
![training accuracies](https://github.com/smhall97/deep_dreaming_music/blob/main/results/training_results.png)

### Dreamed music:
- [Input stft: disco.0 from gztan, optimize difference for classical using subtractive update, reconstruct Green channel ](https://github.com/smhall97/deep_dreaming_music/blob/main/results/disco-subtractive-optimize-for-classical-using-difference-chG.wav)
- [Input stft: reggae.0 from gztan, optimize for metal using subtractive update, reconstruct Blue channel](https://github.com/smhall97/deep_dreaming_music/blob/main/results/reggae-subtractive-optimize-for-metal-chB.wav)
- [Input stft: sinusoidal frequency sweep, optimize for hiphop using additive update, reconstruct Green channel](https://github.com/smhall97/deep_dreaming_music/blob/main/results/stft_sweep_hiphop_chG.wav) 
- [Input stft: sinusoidal frequency sweep, optimize for jazz using additive update, reconstruct Green channel](https://github.com/smhall97/deep_dreaming_music/blob/main/results/stft_sweep_jazz_chG.wav)

#### Visualisation of dreamed jazz
![visualization](https://github.com/smhall97/deep_dreaming_music/blob/main/results/stft_sweep_jazz_sm.gif)

---
## Discussion
In applying the dreaming to the spectrograms, we gain insight into the learned representations the model uses to perform the classification. We are able to visually represent these features, as well as interpret them in audio formats. 

We were limited by our audio reconstruction techniques which immediately revealed a catch-22.

We were able to achieve high classification accuracies when training the models with Mel-spectrograms, but these cannot be cleanly reconverted to audio data.
While STFTs are cleanly converted to audio (without noise added during reconstruction), we achieved lower classification accuracies when training with STFTs. This poor classification accuracy suggests the model didn’t learn useful internal representations that can be maximised during the dreaming process that will transform an input while obeying the laws of spectrograms to ensure it can be reconstructed into audio. 

### Future directions and ideas

The focus on only visual representations of the audio data limited our results as we were limited by audio reconstruction techniques. Future work could investigate using the raw audio signal and using models better suited to signal data (e.g. RNNs, Transformers or autoregressive models such as WaveNet). This limitation was further evident in our use of networks pre-trained on ImageNet alone. Future work could incorporate pre-training on more applicable data such as STFT transforms to help the model learn better internal representations to be maximised during dreaming.  

---
## References
1. Olah C, Mordvintsev A, Schubert L. Feature visualization. Distill. 2017 Nov 7;2(11).
2. 	Google AI Blog: Inceptionism: Going Deeper into Neural Networks [Internet]. [cited 2021 Aug 16]. Available from: https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
3. 	Palanisamy K, Singhania D, Yao A. Rethinking CNN Models for Audio Classification. arXiv. 2020 Jul 22;
4. 	Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:14091556. 2014; 




