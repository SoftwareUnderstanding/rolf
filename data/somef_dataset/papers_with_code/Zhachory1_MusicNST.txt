# MusicNST

Project to combine and experiment with Neural Style Transfer with Music.

## Tests

We test multiple setups:

* NST (with VGG) with MIDI straight into it
* Modified NST (with our own custom model predicting song's artist) with MIDI pushed straight through
* NST with lower dimensional music (sampled from MusicVAE)
* Modified NST with lower dimensional music

## Models
There are a couple of models to keep track of here. I will list them and provide documentation about them here.

* [Neural Style Transfer](https://arxiv.org/pdf/1508.06576.pdf)
* [MusicVAE](https://arxiv.org/pdf/1803.05428.pdf)
* [VGG16](https://arxiv.org/pdf/1409.1556.pdf) (we opted for 16 instead of 19 because why not)
* Custom Model (discussed below)

## Results

![Figure 1](paper/tracks.png)

These tracks are pulled from iterations 1, 10, and 100 from the Neural Style Transfer model using VGG as the backbone without the VAE preprocessing. You can notice that the model is creating denser formats the more iterations it does. We did not tune the parameter for putting a ratio between content and style loss, which was set at 1:4. Tuning this would've put less emphasis on imposing a style in the content music, which could have proved to be better. 

![Figure 2](paper/flatline.png)
![Figure 3](paper/curvedline.png)

Figure 2: Music neural style transfer loss curvature. The x-axis is number of steps and the y-axis is loss value. The one on the left was generated using our simple music classifier, while the one on the right on figure 3 was generated through a VGG model. 

We believe that our convolutional network was too shallow to transfer music style appropriately. When we trained with deeper networks we were overfitting with more layers. However, when we trained using the VAE and the same network architecture we were underfitting. This hints that we need to explore deeper architectures when dealing with music style transfer. 


Given significant processing constraints we were unable to train a good copy of the music VAE. Therefore, we decided to halt Experiment 3 & 4. We were only able to train our VAE for a few epochs which was insufficient, given that when we tried to use it to retrain a better classifier the results were worse than just feeding in midi files straight in. The accuracy on the test set was about 10% lower(accuracy on the simple classifier is 50% and on the one trained with a VAE is 41%). 

![Figure 4](paper/badloss.png)

Figure 4: Loss curve on test and train on the composer classification model trained using VAE as an encoder over the epochs. 

These results from training the composer classifier with latent code from our VAE were inconclusive since the encoding model was not thoroughly trained. Looking at the distribution of all the Z values, no value went passed 0.0025 and clustered around 0. These low and clustered numbers suggest that we trained with too much emphasis on minimizing the KL divergence. Thus, the reconstruction error was shadowed, consequently taking away our ability to reconstruct our input effectively. 

## Conclusions
Further experiments need to be conducted with a better trained Music VAE. We spent a significant amount of time redesigning the VAE in order to overcome the constraints of the one delivered by the Magenta Team. Unfortunately, it was not effective when used in training because we were unable to fully train it. Moreover, our custom classification model showed good results when trained on test data but was not very robust when used for music style transfer straight from midi files. There were so many nuances from converting midi files, to an intermediate format, to a piano roll/tensor format. There were also many types of channels that could've been pulled from MIDI files, such as onset, offset, active, velocities, along with others. These complications made it difficult to move data around and work with.

