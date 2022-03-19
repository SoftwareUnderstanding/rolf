# GestureGAN
### Application of AI in ElectroAcoustic Music

**It should be noted that Google Colab documents run from the Dataset on my Google Drive

Here is a collection of the code which contributes towards the GAN. 

Firstly, the Dataset is provided from a Kaggle competition. User Daisukelab presents a Preprocessed Mel-Spectrogram representation of audio clips, which are in the form of Numpy arrays.

link: https://www.kaggle.com/daisukelab/fat2019_prep_mels1#trn_noisy_best50s.csv

These arrays are depedent on the length of the audio clips. In our case we want these to all be the same length so I wrote some code to unload the pickle and shorten these arrays to be 1 second long. Program I wrote is provided as a file called ReshapeTheNumpy.ipynb.

Original Numpy format: (128, AUDIO_LENGTH, 3);
New Numpy format: (128, 128, 3);

I originally wanted to use AudioSet, however this dataset only provided csv files of timestamps within YouTube videos. These timestamps represented sounds of a particular condition which was ideal for this interpretation. The FAT2019 Prep-Mels Dataset will have to use a classifier engine to organise these sounds. At the moment I plan to break these down into folder of their particular categories and then train the GAN on each. At this current time I am unsure how this will work with the plan for interpolation between categories, but this will soon be fixed.

AudioSet: https://research.google.com/audioset/

To prove that the numpy arrays can be played back into sound, I tried to adapt Daisukelab's code that he used to generate the dataset. In this program he generates a 2D numpy array, but then transforms this to 3D when converting the STFT representation from Mono to Colour. Potentially this could be benefinicial to the acceptance of the dataset with DCGAN. Because this Generative Adversarial Network is built for RGB images anyway, perhaps it is worth just downgrading the output arrays back to Mono so then they can be processed back into sound.

Yet to be succesful in transferring Mel-Specs back to Audio. Might take advantage of this python program that will allow me the ability to create STFT and inverse it.

Tim Sainburg's implementation of kastnerkyle's work: https://timsainburg.com/python-mel-compression-inversion.html

Because of my inability to convert the MelSpec processed dataset back to sound, I took the raw wav files, reduced their lengths to one second and then processe them through Tim Sainburg's implementation. The audio length code I wrote is called AdjustAudioClip and the implementation of kastnerkyle's code is found in the STFT program.

The resulting Dataset I now have is Numpy arrays of 2 dimensions. Now I must try to implement the Generative Adversarial Network model.

The problem with this is that the arrays are not in a shape that could fit into the image GAN. However the MelSpecs that Daisukelab produced before are perfect for this. I will contact him to ask about the possibility of converting these specs back to sound.

Because of my inability to be able to succesfully regenerate audio from the STFT images I recieve from Daisukelab's code, I returned to the GANSynth code. From reading their papers before, I understand that they are fully able to perform Audio->Spec-> Audio tasks. Inside their Python program, Specgrams_Helper, they define function's named "waves_to_specgrams" and "specgrams_to_waves". I will look into these methods soon.

//THE GENERATIVE ADVERSARIAL NETWORK MODEL THAT I HAVE PREVIOUSLY EXPLORED//

The model is built upon a version of DC-GAN outlined by lecturer Jeff Heaton.

DC-GAN with Images: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb

This application uses images which are represented as 3 dimensional numpy arrays. To tackle this we must adapt this model to accept 1 dimensional arrays that are outputted from our STFT.

![GeneratedImages](https://github.com/djcustard/GestureGAN/blob/master/Images/ADevelopedImage.png)

<i> These were some numpy arrays creating audio that I generated, here displayed as images. These were all noise.</i>

Currently following a 1D function implementation GAN. Keras model Sequential first layer must take it's input shape. Use Dense function to define this.

How to Develop A 1D Function Generative Adversarial Network: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/

Following on from this, I am trying to develop a short term version of this system to produce waveforms. Replacing the 1D function with a waveform. This work has seen me starting to train a discriminator model. This taking various points from the real waveform and a randomly generated waveform (noise.)

This interpretation of a 1DGAN is currently training on a waveform. Because of the complexity of the waveform, the discriminator model usually trains to a 24% recognition of the real signal and 100-99% recognition of fake. I imagine this wil have a serious effect on the ability to replicate the signal. I am first seeking to try and make sound, then to enhance the models ability of production.

Signals that are produced from the 1DsGAN are completely silent at the moment. But today I will work on this more.

After some testing, the value predicted from the Generator model is corresponding to how large the latent space is through its layers. I believe I have been training the Discriminator model on too little amount of layers. With the complexity of the signal, I am training a model with an increased amount of layers.

Also, I could be improving my performance by choosing 44100 random samples at each level to try and understand the signal better. I currently only train on a 22050 batch as 44100 points are split between 22050 real and fake samples. --> This did not improve things.

However 1DGAN kernel initializers were previously set to he_uniform, which means value were accustomed to a normal distribution around 0(link1). This shouldn't be as such as we are dealing with floating points with a random uniform distribution. Moving back and across over 0 (in terms of audio). Just by changing the layers from he_uniform to random_uniform, the discrimintor model saw a boost in performance registering recognition of real samples by 49% and real by 100-99%.

Currently the 1DGAN works off the function being of symettrical uniform. Random points are selected from the real function and these each connect to form a rough outline of the functions shape. However I propose that we take 'slices' rather than points from a signal. This would mean that 'slices' of continuous audio are fed into the discriminator model, rather than individual points. Because of the complexity of the signal from the audio clip, taking 'slices' should furthermore give the model and understanding of how the points are arranged and positioned throughout the clip.

02/04/20 - DsExGAN is the python program I have interpretted this idea into. I will upload the code soon, currently trying to train the discriminator. DaisukeLab helpfully responded to me today saying that I should check out the Librosa function or potentially research a GAN model that creates plausible sounds based on the Spec image.

03/04/20 - DsExGAN has been trained and tested. The GAN seems to output values that are more realistic. For curiosity I will train a model on much less layers to see its success. I have also multiplied the Output of the GAN files as these are still float values between 0.0-1.0. We will multiply these by 32767 to potentially change the behaviour. I am looking through TensorFlow's own tutorial on custom training. Very similar to 1DGAN lesson, will practice storing audio into Tensors rather than Numpy's. Whilst I am trying to drive the success of getting audio which is like that of the sample, I am also hoping to improve my Discriminators performance by building a more precise model. I am unsure on how the success of multiplying the output will be, however after this I will change the output layer of the Generator Model.

'<i>The ideal number of hidden layers and neurons depends on the problem and the dataset. Like many aspects of machine learning, picking the best shape of the neural network requires a mixture of knowledge and experimentation. As a rule of thumb, increasing the number of hidden layers and neurons typically creates a more powerful model, which requires more data to train effectively.</i>' - Custom Training: Walkthrough (TF)

06/04/20 - Been transforming Numpy arrays to Tensors. Will focus on building a new model today. Training will be overnight. To build this new model, I will be creating a big Numpy array, transferring this to a tensor then transferring this to a tensor dataset. So far today I have created a Dataset from a single audio clip and have built a generator model! Taking influence from the DCGAN model from TensorFlows tutorials, I have thought about returning to generating image from audio. To make this succesful I will need to resize some WavFiles (again, doh!) If I trim my clips to 22500 sample, they will have a tensor shape of (22500, 1), I will then reshape these to a format of (150, 150, 1). This technically being a greyscale image. From the DCGAN[link4] implementation on TensorFlow, I have created a generator which will produce images of this format. My method now is to implement the tutorials architecture, and verify the success of the models. This will be done through plotting the gradient and optimizer performance. For now, I will reshape 1 audio clip to the specified length and save this tensor.

07/04/20 - Deciding to reshape a batch of audio clips. Because of the bigger size of the images, there is much more processing needed in the Convulotion of the images. This produces some errors. I will return to GANSynth to understand the Tensors they produced with their STFT processing algorithms. It's important to note that the images I am trying to create represent the raw waveform. No STFT processing has occured on these audio clips as of yet. However recently I have believed that there is potential to do this without producing an STFT image. We will continue with raw data for now. Scaling of audio clips going into DCGAN and out may need to be scaled better. We must scale the audio around (1,-1) to fit the GANs specification. The output from the network will then be scaled back into its file preferance. Before doing scaling I found these results:

<p align="center">
<img width="300" height="300" src="https://github.com/djcustard/GestureGAN/blob/master/Images/DesiredResult.png">
</p>

<p align="center">
  <i> This was the Image that was used as the Dataset. Here is the sound</i>
</p>

<p align ="center">
  https://soundcloud.com/comeheretohavearave/datasetsound24
</p>


![GeneratedImagesEpoch1](https://github.com/djcustard/GestureGAN/blob/master/Images/1.png)
![GeneratedImagesEpoch50](https://github.com/djcustard/GestureGAN/blob/master/Images/50.png)

<i> Epoch 1 compared to Epoch 50. My stride and processing is at too low of a level for (1, 150, 150, 1). This develops this grid like texture. The Audio for these clips will be added soon.</i>

As mentioned, because of the smaller strides, we recieve a grid like texture which changes in gradient over the epochs. This is evident in my results because I have tried to apply an architecture for smaller resolution images, on a much larger image. I will experiment with the generator and discriminator networks, adjusting the scale of convoluting throughout. Any remarkable results will be posted.

08/04/20 - A lot of today has been about functionally developing the project. Google Drive was running out of space so just moving information to my HD. Because of the behaviour witnessed in yesterdays GAN production, I believe that DCGAN's architecture won't be able to process the necessary amount of information. On research, Progressive GAN architecture could fit our particular needs. Supporting the upscaling of images to the desired output. (This is the architecture of GANSynth.) For the rest of today I will focus on reading through the paper[link5] that establishes the concepts of PGAN and then read through some Machine Learning Mastery posts on the subject. [link6] [link7]

09/04/20 - Building a PGAN today. I think I will reconstruct a dataset appropriate for this model and DCGAN too. I will retry DCGAN with this new dataset and begin to recieve results from PGAN. Each of the 100 resized audio clips will be appended in a Numpy array. These will then be transferred a tensor and reshaped to a 10x10 grid. I'm also investigating representing the signals as 3 dimensional images (RGB).

Dataset is underway now. We are currently forming one with the shape appropriate for smaller PGAN work; tensors with shape (1, 128, 128, 3). First these are trimmed to wav's of 49,152 sample length. Then these will be converted to tensors. Here are some of the images:

![RGBSOUND1](https://github.com/djcustard/GestureGAN/blob/master/Images/1RGB.png)
![RGBSOUND2](https://github.com/djcustard/GestureGAN/blob/master/Images/2.png)
![RGBSOUND3](https://github.com/djcustard/GestureGAN/blob/master/Images/3.png)
![RGBSOUND4](https://github.com/djcustard/GestureGAN/blob/master/Images/5.png)

<p align='center'>
<i> These are plotted RGB tensors. I will add the sound for each of these so they can be heard. </i>
</p>

Already we can see that the detail is very finite in these images. We will see whether the model can replicate these traits.


It its important to note that PGAN have the potential to create 1024x1024 size images. Whilst this takes considerable power and will most likely involve renting time on a higher spec computer. This could be a possibility. If the PGAN works on a smaller scale with random audio clips, the next steps will be to:

- Develop a classifier and organise the audio clips into folders of their specification.
- Attempt to create a combine a conditional network design (introduce controlled interpolation)
- Create higher resolution, and in-turn, longer audio clips.

10/04/20 - Still following tutorial on the construction of P-GAN's. If I am able to get results from this network. We will then begin inputting audio. Loading in this dataset has taken a huge amount of time. Because of the 20k images being loaded in, this has taken around 5 hours. After finalising the arrangement of the dataset for the google colab. The session no crashes after using too much RAM. I will try this again tomorrow on different settings, but I will then also try and learn how to implement the code on my own system. Perhaps I could run the code and get results locally.

11/04/20 - First run of PGAN code of the day. The session crashed after using too much RAM. However I was offered the ability to connect to a notebook with more RAM, hopefully this should improve my opportunity of results. I have started to recieve results! This is from the practice dataset which is the Large CelebA set[link8]. This means I will then be able to input the plots of sound. The dataset I currently have prepared has a size of 3800 clips. I will add clips from other folders to then have this figure to a possible 20k. I am also considering investigating whether I can increase the visual range of these images by increasing the numpy values by 3x. Hopefully there will be a larger range and more contrast in the plots. I will try this today and post results here.
<p align='center'>
<img width="500" height="400" src="128x128FACES.png">
</p>

These are faces generated from the PGAN model from the tutorial[link7]. Tomorrow I will train the audio dataset on this architecture. Before this I will complete a few last tasks today:

- Adjust audio clip length and ensure a larger amount of uncurated samples are available.
- Experiment with representation of clips (3x float values to increase range)
- Ensure clips retain their properties when transferred back to sound
- P-GAN code to include GIF representation of total epoch's. This will be incorporated from a TensorFlow tutorial. [link9]
- Plotted representation will feature 3 images that are larger
- Audio clips and plots will be saved more frequently

12/04/20 - Audio clip dataset now totals 26860. These are uncurated and unspecialised. Potentially these may not produce any interesting results from the netowrk. However we can improve the dataset at a later time if need be. Curating the dataset will mean building a classifier and categorising the audio into sub-folders.

13/04/20 - Updating the code for PGAN and the incoming dataset. I will then train this model and produce audio. I have yet to implement the GIF representation, however loading in the dataset will take nearly 7 hours. Luckily this will be saved as a numpy array after. It is now at a predicted 16 hour period to load in the audio.

14/04/20 - Dataset load in complete, now training the model. Apparently I am unable to do this as I have used up my allocated slots. I will try to connect locally to my system. I trained this model and recieved dissapointing results. Here are the images that were generated:

![FAIL1](https://github.com/djcustard/GestureGAN/blob/master/Images/FAIL1.png)
![FAIL2](https://github.com/djcustard/GestureGAN/blob/master/Images/FAIL2.png)

These are from two different stages of the PGAN process.

15/04/20 - After training the model. The images recieved were very disappointing, I believe because the traits of these images are very detailed. (Rather a texture than a face) The model was unable to fully understsand these through Convolution. Today I will tak a different approach and use SciPys STFT method. Potentially, because the contrast in these plots is much greater, we could imagine them to allow the model to train much better. SciPy's STFT can generate small numpy arrays of (223,223) but of dtype complex128. Whilst looking through this however, I realised that my dataset had become distorted, potentially this could be a reason for the fault in my dataset training. Listening back to the clips as well, they are sped up. I believe this is to do with the incorrect Sample Rate on the wavfile. I will try to ensure this doesn't happen again and we will retrain. Because of the lack of results thus far in these processes, I am finding myself hard to motivate. However, we'll see if this training on the model presents any interesting results. Here is an image representation of the distortion:

<p align='center'>
<img width="500" height="500" src="Images/distorted.png">
</p>

16/04/20 - Distortion in the images was always there. When I had been displaying the audio, I was using a different format with the numpy arrays. Today I'm working on solving the image representation of the audio clips. Referring back to Daisukelab's comment on the preprocessed Mel Spectrogram dataset, it was mentioned that a GAN could be made to try and generate audio from spectrogram image representations. Yesterday I did some research on this and came across a multitude of groups that have researched into this area. TiFGAN[link10] seems to generate and then recreate audio from STFT plots. I will spend more time looking over this work. Producing audio from the plots is key as this will be what is fed into the PGAN. Looking back at my previous distorted plots, because they have each been recorded at the wrong sample rate, they now contain a lot less sample points, I am looking into scaling these image representations to the (128,128) format. Scaling could potentially increase definition of features on a plot.

In a very surprised turn of events, my PGAN now works! The dataset may need to be readjusted due to the wrong sampling rate, however this can be done after we hear the first audio clips. Here are the plots so far, I presume the model will complete around 19:00:

![TRAIN1](https://github.com/djcustard/GestureGAN/blob/master/Images/TRAIN1.png)
![TRAIN2](https://github.com/djcustard/GestureGAN/blob/master/Images/TRAIN2.png)

After the model has completed training, I will be able to seperate the Generator and produce more audio clips. I will produce a folder of samples. From here I will now need to build a classifier and begin splitting the audio files into there classification. This also allows be to begin working on my composition. I think it's important for me to further adjust the PGAN architecture.

At lower resolutions, I will scale the array up to the size (128,128). This is so that sounds can be heard to shape closer to a classifier sound. For example if we are training the model on sounds of guitars, because PGAN is training from the lowest resolution of (4,4) and increasing, we want to scale these arrays to dynamically hear the changes and the Network trains.

17/04/20 - After the success with the model yesterday, unfortunately it was unabel to save the last 128x128 tuned layer of the model. This is the key finished version so I must train up the model again today. I added a line in the code last night which allowed the model to produce audio from the plots, however I believe this could have made it unstable so I removed this code. Once the PGAN model is fully trained, I will be able to extract the generator and produce audio clips. Fingers crossed for sound today!

18/04/20 - I have trained the model for two days in a row for the past two days, however on the last stage the image nor the model is saved. I believe that I run out of my allocated Google Colab time. Because of this I will save he discriminator models today as well whilst it trains. Then tomorrow I will train the final stage seperately. Today we'll be able to listen to the fadedd 128x128 model.

19/04/20 - Training yesterday and today has produced some rather odd results. They give the impression to me that if I save the discrimintor then it resets? It doesn't seem to catch behaviour as it used to. For comparison I will show plots before and after saving was implemented:

21/04/20 - The generator's produce sound, however not ones that I want. To try and fix this, I'm going to return to our visual representation of audio. I think there are a few errors here, such as the distortion. Reffering back to the image representations, we want this to be scaled around 1,-1. This allows for better processing in the GAN network. At the moment, these are the values it is scaled around:

I will find out the range at which these cover through research. These are 32Bit Depth audio clips. Their values lie between the range of -2,147,483,648 and +2,147,483,647. In imshow, the values want to range between 0,1. Today I'll create a new dataset which has divided by the max 2,147,483,648 to try and adjust to the [-1,1] scale. Potentially I will be able to get some better results then.

Whilst I build this new dataset, I will try to troubleshoot a few different ideas. Try and approach the project from a different angle. Potentially go straight ahead and build a classifier network. I will go through this post [link11]. Going throught this post, the convolution is expecting RGB values for a convolution. An RGB value is created out of 3 0-255 values. The problem with this is that currently I have a numpy format like that of an image (128,128,3). However these have values in the range defined previously. If however I have an audioclip which is of 16,384 samples, then I could have each 3 bits to define a single sample. With a sample rate of 22050, this would be just under a second in length. Before I adjust the length of the audio clips again, I will return to the stft to assess the shape of arrays I recieve there. Because 16,384 samples is not that long, potentially STFT could provide longer. I think I will also do research into Amazon's renting of notebooks as I may need a higher level processing unit that I have on access at more times. If I was to have this capacity I could upgrade the code to generate images of (1024,1024,3) this would then be samples of 1,048,576. With a sample rate of 22050, this would be 48 seconds. This is too far into the future to fully conceive, first I will generate images to try and see what they look like.The proposed idea to have an RGB value represent a single sample is a hard idea to implement. It means that I will have to write a code that takes the floating point value of the sample and represent it by 3 0-255 values. Perhaps I will lose resolution as I wont be able to fully arrive back at the same floating point.

At the moment, with a 32 bit depth, there is a huge range. Just by taking the max value 2,147,483,648 and cube rooting, we recieve a value of 1290.0... If we were to even times all max values of RGB, then we would be nowhere near our target. To work around this, we coul work at a 16 bit depth rate which has a range of -32,768 and +32,767. This range could easily be covered and represented by RGB values. I will present a conversion of the same audio clip to have evidence of the reduction.

Because of my capabilities in Java (and because I can't run another notebook in Google Colab) I have been working on ways to reduce each 16bit depth sample to an RGB value. I have a method, however it reduceds the range to 23,320. This is still a great range than a 12bit depth. Tomorrow we will hear and evaluate the change.

22/04/20 - Overnight, I have considered just creating images of (256,256,3). This is just because I'm not sure whether audio clips will be recognisable in such short time. This will give us audio clips of length 65,536 sample points, over a second long. Audio clips are now successfully converted into RGB representations and can be returned to audio. Audible difference is minor. Here are some plots:

![RGB1](https://github.com/djcustard/GestureGAN/blob/master/Images/download_(3).png)
![RGB2](https://github.com/djcustard/GestureGAN/blob/master/Images/download_(4).png)
![RGB3](https://github.com/djcustard/GestureGAN/blob/master/Images/download_(5).png)
![RGB4](https://github.com/djcustard/GestureGAN/blob/master/Images/download_(6).png)

24/04/20 - The past two days have been spent trying to produce the dataset. This has given me issues due to its size (256,256,3). I have tried producing this dataset with 20k samples, however I am now settling for 10k. Because I have run out of my GPU access for today, I will produce a 20k size dataset of samples of (128,128,3). These will be under a second, however will provide a safety net incase the (256,256,3) provide too computationally expensive in the network. Producing a new dataset of 16,384 sample point audio clips, I reduce the Sample rate to 22.05kHz as well. This will slow the original clip by half the speed.

26/04/20 - Since my last post I have been trying to create a dataset. Because of the size of nealry 20k audio clips, it was hugely RAM intsensive. Thus I ended up producing a 7.5k sized dataset with (256,256,3). Today has been the first day I've been able to train far into the process. I will post some images later to show progress. As I am unsure about how succesful this representation of audio will be, I will do more research into STFT.

27/04/20 - Today I have spent the day setting up a new Progressive GAN architecture which I will then combine with a Conditional architecture tomorrow.

28/04/20 - Working on Classifier today, just started it working. The baseline model comes from the Fashion MNIST Tensorflow tutorial. Because I have yet to fully develop the model and the Fashion MNIST works with resolutions of (28,28,1), I will be developing the Classification model myself making it suitable to the Dataset. There are 81 possible tags that the classifier can class the audio into. Currently the model trains to a not very impressive performance. See below:

![ClassifierModelTrain](https://github.com/djcustard/GestureGAN/blob/master/Images/badModel.png)

As we see loss is much greater and our accuracy is only about 14%, by lunchtime (a very childlike goal post) this will hopefully be above 50%. Under my knowledge, I will create a Sequential Model that will use two or three layers of Convolution, then I will output a value to represent the tag. 

I have developed a classifier that trains to 70%+, evaluating the model on the noisy dataset will allow me to see whether the model has overfit. Yet to test the noisy dataset, I had some problems unzipping so uploading to Google Drive a raw pickled version. Find below the results from the train dataset:

![ClassifierGoodModelTrain](https://github.com/djcustard/GestureGAN/blob/master/Images/goodModel.png)

The Noisy Dataset will supposedly take 6 hours to upload to Google Drive. I have also been testing the new PGAN model that I tried to implement yesterday, this failed to show any sign of producing promising results. Tomorrow I will look at introducing the Conditional elements after I test the model on the noisy dataset. The dataset will be split into different the folders of the different labels. Now the upload shows a more reasonable time of 1 hour and 47 minutes.

29/04/20 - Unfortunately the classifier model overfit with the test data only having an accuracy of 9%. Today I will reduce the complexity of my model and increase the dataset it trains on to reduce overfitting.

![ClassifierOverfitTrain](https://github.com/djcustard/GestureGAN/blob/master/Images/overfit.png)

30/04/20 - Today I am readjusting the PGAN model to Convolute with a horizontal priority. Previously the model had convolutional layers with kernel size 1x1, 2x2 and 4x4 which meant that convolution occured in a square format. This is appropriate for normal images as there is no dimensionality over the 2D images. However in the case of the audio representaion, the signal is represented from top left to bottom right, moving the floating point values over RGB values in a horizontal pattern. This is evident from the representations above. I will train this model and also work on my own model. The intensity of possibility for error in these cases is very high. For my classifier model, I am reading the _Going Deeper With Convolutions_ paper by Szegedy(2014) et al.[link13]

I believe I have changed the wrong dimensionality on the convolution layers. I will review the 16x16 tuned layer, if this doesn't seem correct I will switch the values provided. The convolution layers have kernel window sizes of 1x1, 6x1 and 8x2.

I am experimenting with the success of my classifier currently. By making slight adjustments to the layers, I am hoping to find which method improves the accuracy best.

31/04/20 - Because of the lack of coherent sound so far, I will begin to produce a composition to demonstrate the potential of these models. I will consider how motion and energy are applied into sound. I am still testing the classifier too.

01/05/20 - 02/05/20 - I have established that my gestures will fall into 4 categories which summarise the enegy output or, rather so, the envelope of the sound. These titles are as such Trigger, Trigger Continuous, Continuous Variation and Continuous Repetition. The titles define the energy behaviour across the acoustic bodies and thus their sonic profile. Find below the defintions for these:

- Trigger: A sound which is heard after a summation of energy has peaked above a threshold. This causes a sudden change of state.

- Trigger Continous: Energy is applied to the acoustic body. This produces an initial peak and either the energy still moves within the body, or their are smaller amounts of energy still being applied to the body.

- Continous Variation: A stream of energy which varies in intensity from the listeners perspective.

- Repetitious Variation: A sound which is held or continuously loops.

Alongside these definitions I have collected sounds which demonstrates each categories definition. This collection of sounds will be submitted as a dataset that could potentially be used for neural network training in the future.

03/05/20 - I generated a score for the composition. The material outlined expresses the different gestures heard overtime and the eventual introduction of interpolated gestures.

04/05/20 - Following the score produced from yesterday I began to compose the composition with the dataset sounds. The piece beings with distorted clips of audio. This is to represent the first stage of Progressive GAN's image building process. Generating at a small 4x4 resolution.

05/05/20 - The composition develops from

06/05/20 - Composition draft complete

07/05/20 - DATA REPRESENTATION:

I am returning to the problem of data representation whilst commenting my code. From the knowledge I have learnt, I understand that for models to understand the images they recieve and map their input, contrast and definition is key. 

17/05/10 - I have been working on the survey today. Generating the videos to 

Links:
<p>link1 - https://keras.io/initializers/</p>
<p>link2 - https://www.tensorflow.org/tutorials/customization/custom_training</p>
<p>link3 - https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/</p>
<p>link4 - https://arxiv.org/pdf/1511.06434.pdf</p>
<p>link5 - https://arxiv.org/pdf/1710.10196.pdf</p>
<p>link6 - https://machinelearningmastery.com/introduction-to-progressive-growing-generative-adversarial-networks/</p>
<p>link7 - https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/</p>
<p>link8 - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html</p>
<p>link9 - https://www.tensorflow.org/tutorials/generative/dcgan</p>
<p>link10 - https://tifgan.github.io</p>
<p>link11 - https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/</p>
<p>link 12 - https://www.tensorflow.org/tutorials/keras/classification </p>
<p>link 13 - https://arxiv.org/pdf/1409.4842.pdf </p>
