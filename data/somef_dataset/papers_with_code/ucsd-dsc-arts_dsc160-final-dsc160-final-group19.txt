# Project Title

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Nikolas Racelis-Russell, nracelis@ucsd.edu
- Iakov Vasilyev, ivasilie@ucsd.edu
- Cameron Shaw, c8shaw@ucsd.edu

## Abstract

  Generative music is not a new idea, and has been around as early as 1989. However, the use of neural networks for this creation has not been popularized until recently (Yang et al., 2017). This project plans to use neural networks to generate music from MIDI jazz files. However this presents many challenges, as melody can be generated, but structure is much harder to generate due to variance. We wanted to compose music using neural nets to attempt to generate melody from MIDI files scraped off the internet.
  
  Right now even the best computer-generated music is not good enough to be considered an actual source of entertainment, and the examples that come close are usually heavily stylized compositions. In the future, however, there is a chance that machine-produced entertainment will rival that of human origins, and we tried to see how close we can get to that point with the current algorithms and levels of technology.

## Data and Model

(10 points) 

__Models__:
- [Our WaveNet Adaptation (There's 3 different versions we tested)](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/tree/master/code/WaveNet). We adapted the traditional WaveNet architecture, which uses raw waveform data, and changed it to accept string sequences of integers representing notes. The model uses 1D Dilated Causal Convolutional Layers. The most important aspect about this is the dilation, which covers for the low receptive field of the convolutions by exponentially increasing it inside the hidden layers.
  - [WaveNet Paper](https://arxiv.org/pdf/1609.03499.pdf?utm_source=Sailthru&utm_medium=email&utm_campaign=Uncubed%20Entry%20%2361%20-%20April%203%2C%202019&utm_term=entry)
  
![WaveNet Architecture](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/git_img/wavenet_architecture.png)
![dilation](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/git_img/diluted_causal_CNN.gif)

- [GANs](code/GANs.zip). The idea of a GAN is to have a generator model creating data and a discriminator model classifying it as real/fake trying to outsmart each other. This directly speaks to our attempts at making the music human-like, after all, real/fake classification is the means and the goal. GANs have been shown to produce incredible results in image generation, however, traditional GANs struggle with data directionality, which is one of the main features of music generation. Therefore, the descriminator has to rely on sequential neural networks, usually LSTMs for discrimination. We did not get any satisfying results with simpler GAN models, so our theory was that the generator does not get enough attention. An example of a well-built GAN model is MuseGAN, which utilizes three different approaches to note generation, as well as a layer of bar generation. Sadly, we could not train it on our data as the preprocessing was done very specifically for the dataset the creators used.
  - [MuseGAN paper](https://arxiv.org/pdf/1709.06298.pdf)
  
![MuseGAN Structure](git_img/musegan.png)

- [Performance RNN](https://github.com/magenta/magenta/tree/master/magenta/models/performance_rnn). Magenta's performance RNN uses a LSTM, Long-Short Term Memory recurrent neural network. The point of which is to retain the memory from previous training steps of the neural network and to use those steps later on in the line, something that some other neural networks lack. The performance RNN models notes in a similar way as a midi file. It represents notes with a starting pitch and ending pitch event, a time shift event, and velocity. These events are used to represent a note and its particular dynamics for modeling.

__Training Data__:
- [Maestro Dataset](https://magenta.tensorflow.org/datasets/maestro). A dataset released by Magenta that has over 200 hours of piano music, and is in midi format.
  - [Maestro sample 1](/data/SAMPLES/maestro_sample.mp3)
- [Video game midis](https://www.vgmusic.com/music/other/miscellaneous/piano/). A bunch of only piano midi files
  - [Video game sample 1](/data/SAMPLES/ff9_battle.mp3)
- [Schubert](https://drive.google.com/file/d/1qnQVK17DNVkU19MgVA4Vg88zRDvwCRXw/view). Random piano midi dataset
  - [Schubert sample 1](/data/SAMPLES/schubert_sample.mp3)

## Code

(20 points)

__WaveNet__: 
- [Scraping video game music](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/code/WaveNet/Scraping%20Video%20Game%20Music.ipynb) Crawls a link for more download links found on page.
- [Base Model](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/code/WaveNet/WaveNet_midiV1.ipynb). First iteration of the model, covers processing of midi files, and then runs baseline WaveNet Model originally ran on maestro dataset.
- [WaveNet v2](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/code/WaveNet/WaveNet_midiV2.ipynb). Second iteration of the model, this time adds in removal of notes occuring less than X times, and also changes hyper parameters of the model in an attempt to fix generative process. Trained on videogame music.
- [WaveNet Mini](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/code/WaveNet/WaveNetV2_small_videogame.ipynb) Failed experiment where we tried to see if having a very small dataset that would become overfit would produce decent results.

__GANs__:
- [Models](code/GANs.zip). The two GAN Models are in the GANs.zip folder. The Untitled notebook in the midi-lstm-gan subfolder can be run to train on the Maestro dataset.


__Magenta's performance_rnn__: 
- [PerformanceRNN](code/magenta_creation_code.txt). In order to properly set up the magenta environment, it was set up according to their qualifications [here](https://github.com/magenta/magenta). This code was then run once the environment was complete. 

## Results

(30 points) 

__WaveNet__:
- [Version 1 (maestro) Sample](/results/WaveNet/test_1.mp3). In this version we can melody forming, but at the time this was the first sample that wasn't just the same note over and over again. So to remedy this we wanted to try a new dataset and change the model from its baseline, which is where the next sample comes in.
-[Version 1 (schubert failed sample](/results/WaveNet/schubert1.mp3). From trying out the model trained on a very small data set named schubert, but not much good came from this model, barely any nice melody from the samples.
- [Version 2 (video game) Sample](/results/WaveNet/videogame_1.mp3). With this sample we can hear the heavy videogame music influence, as it sounds kind of similar to some Final Fantasy title screen music (about 20 out of 800ish data samples were from final fantasy). This was a definite improvement from the base model, possibly due to the generative process changes from tuning hyperparameters [linked from this code notebook](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group19/blob/master/code/WaveNet/WaveNet_midiV2.ipynb).

__Muse_GAN__:
- [midi-lstm-gan after 3000 epochs](results/gan_final.mid). A constant stream of almost random notes, this model seemed to perform the worst out of all the other ones, the main reason being the music21 midi parser failing to parse the Maestro files correctly.
- [midi-lstm-gan loss per epoch graph](results/GAN_Loss_per_Epoch_final.png). This graph shows the convergence of the discriminator and generator losses at a somewhat high number (x-axis is 100s of epochs), signifying, most likely, underfit data. 

__Magenta's performance_rnn__:

- [first run after 300 steps, on a fraction of the dataset](results/rnn_example1_worst). An incoherent stream of notes following the seed stream of notes. Not only was the model trained on only a fraction of the dataset, it was only trained for 300 steps. Like a monkey slamming it's hands on the keyboard.

- [second run after 3000 steps, on the same dataset](results/rnn_example2_better). Still incoherent, but the timing and musicality is a little more recognizably musical. The monkey listened to 48 hours worth of Mozart, and now thinks itself an artist.

- [third run using Magenta's pre-trained model](results/rnn_example3_best). Nothing close to being a real song, but the timing and musicality is close. The note progression needs work, but it is much better. Our monkey has been going to classes for the past 3 months, and this is its first recital.

## Discussion

  Advancements in the field of generative art have been quite spectacular. With music, the best advancements were made by applying specific models to specific datasets, and, as our experience shows, the models malfunction when presented with music of less structure or different genres. The reason for that, we believe, is that our models ended up underfit. The algorithms tried to extract patterns from many (but not too many) very different samples, and, limited by the memory and power of our PCs, we were definitely not feeding them enough data for any clear patterns to solidify. Of course, one solution would be to give the models even less data, which would cause overfitting. Specifically for this task, overfitting would mean the algorithms would produce very similar music to the chosen set which would make it sound good, however, would it even be new music at that point? Immitating and following rules are tasks that computers excel at anyway, so there is no point in having nice sounding results if they come from overfitting. To avoid both underfitting and overfitting we would have had to feed way more data to the models than we possibly could.
  Trying different models shed some light on the performance issues and advantages of certain underlying neural nets. The three most popular algorithms in music generation turned out to be LSTMs, GANs, and Encoder-Decoder networks. LSTMs seem to be the most widely used networks as they capture both short and long term dependencies very well, which is important for music as there needs to be consistency not only withing each bar, but also between bars withing phrases, as well as between the phrases themselves. Furthermore, music often relies on set verse-chorus structures, and, hopefully, LSTMs can take care of that as well. GANs and Encoder-Decoders usually play secondary roles in music generation, but are also quite useful and important for more advanced models.
  
  So what would an ideal music generative model look like? The ideal model would encompass all possible differences of music, and it is hard to tell what we are currently lacking. On the model side, it would be perfect if we could utilize some type of a “music theory of everything”, although the point of applying neural networks to this task is to let the models figure that theory out.
  Therefore, the problem is on the data side, in the data itself and the way it is processed. The idea is human listeners rely on more than note information for music appreciation, for example we expect different instruments to play different parts, different genres to have different structures, and so on. On top of that, the idea of “enjoyment” is unquantifiable, so there seems to be no real measure of how well a model is doing without a human supervisor, and even then the opinion is subjective. Ideally, we would have compact data that would encompass a lot of information, including (but not limited to): note info (midi covers that pretty nicely), bar info, instrument info, genre info, and maybe even some type of sentiment info. The good news is all of these measurements are achievable to some capacity, and a GAN discriminator could assume the role of an objective human supervisor, so we believe that it would be possible to create a near-perfect dataset and train a near-perfect model that could create catchy human-like music for everyone to enjoy.

## Team Roles

- Nikolas Racelis Russell: WaveNet model
- Iakov Vasilyev: Muse_GAN model
- Cameron Shaw: Magenta performance_RNN model

## Technical Notes and Dependencies

__For WaveNet and GANs__:
- Tensorflow-gpu (latest version as of 6/11/20)
- music_21
- scikit-learn
- numpy

__For Magenta__(requires many different packages but mainly):
-tensorflow
most dependencies are taken care of after running the environment set up. 


## Reference

- __MidiNET, CNN-GAN__: https://opus.lib.uts.edu.au/bitstream/10453/6862/1/2004001187.pdf
- __Combining theory between RNN and LSTM__: http://cs229.stanford.edu/proj2016/report/Lou-MusicGenerationUsingNeuralNetworks-report.pdf
- __Easy to understand example__: https://medium.com/@leesurkis/how-to-generate-techno-music-using-deep-learning-17c06910e1b3
- __Videogame music with one instrument (lots of music theory)__: https://www.youtube.com/watch?v=UWxfnNXlVy8&feature=youtu.be
- __DeepMusic__: https://github.com/llSourcell/How_to_generate_music_in_tensorflow_LIVE
- __DeepBach__: https://arxiv.org/abs/1612.01010
- __WaveNet Paper__: https://arxiv.org/pdf/1609.03499.pdf?utm_source=Sailthru&utm_medium=email&utm_campaign=Uncubed%20Entry%20%2361%20-%20April%203%2C%202019&utm_term=entry
- __WaveNet Architecture Adaption__: https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
- __Videogame MIDIS__: https://www.vgmusic.com/
- __Scraping code__: https://github.com/x4nth055/pythoncode-tutorials/tree/master/web-scraping/link-extractor
- __Magenta repository__:
https://github.com/magenta/magenta
- __Magenta project homepage__:
https://magenta.tensorflow.org/
- __Maestro dataset__:
https://magenta.tensorflow.org/datasets/maestro
