# DoodleGAN

"Quick, Draw!" was released as an experimental game by Google to educate the public in a playful way about how AI works. The game prompts users to draw an image depicting a certain category, such as ”banana,” “table,” etc. Consequently, more than 1B drawings were generated, of which a subset was publicly released as the basis for this competition’s training set. That subset contains 50M drawings encompassing 340 label categories. (https://www.kaggle.com/c/quickdraw-doodle-recognition)

Convolutional Neural Networks (CNNs) have continually demonstrated their superiority over other modeling choices for image based datasets. Since this project we will be dealing with image classification, using deep learning tools like CNNs is a natural choice. Furthermore, in recent years, Generative Adversarial Networks (GANs) have shown to be extremely effective at modeling the distribution of the data.  (http://papers.nips.cc/paper/5423-generative-adversarial-nets) The modeled distribution can be used to generate samples of the data that closely resemble real/true data. We seek to train a GAN to learn how to draw doodles with a given label, i.e the output is an image from a certain class. If we manage to train a good enough generator we could use it as a data augmentation tool that could help to train an even better classifier.

![bee](https://storage.googleapis.com/kaggle-media/competitions/quickdraw/what-does-a-bee-look-like-1.png)



# Simple GAN

Simple_GAN.ipynb contains a simple and somewhat naive implementation of a GAN.  The class of doodles we chose to use are cats (go figure).  An randomly chosen example cat image:

![cat](https://user-images.githubusercontent.com/14242505/50039091-fef49880-ffe0-11e8-8e89-0e17b910cfeb.png)


The generator is composed of 5 dense layers separated by LeakyReLUs.  The descriminator is similar, except it has 4 dense layers.  The architecture is shown below.

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1024)              803840    
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800    
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 1,460,225
Trainable params: 1,460,225
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 256)               25856     
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 512)               131584    
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 512)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 1024)              525312    
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 1024)              0         
_________________________________________________________________
dense_8 (Dense)              (None, 2048)              2099200   
_________________________________________________________________
leaky_re_lu_7 (LeakyReLU)    (None, 2048)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 784)               1606416   
=================================================================
Total params: 4,388,368
Trainable params: 4,388,368
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 100)               0         
_________________________________________________________________
model_2 (Model)              (None, 784)               4388368   
_________________________________________________________________
model_1 (Model)              (None, 1)                 1460225   
=================================================================
Total params: 5,848,593
Trainable params: 4,388,368
Non-trainable params: 1,460,225
_________________________________________________________________
```

Initially, the generated cat images aren't very good.

![simple_bad](https://user-images.githubusercontent.com/14242505/50039096-12076880-ffe1-11e8-90d9-082f49ab66a2.png)

After many epochs of training, they get much better.

![simple_good](https://user-images.githubusercontent.com/14242505/50039097-16cc1c80-ffe1-11e8-92f4-18b94e04c9c5.png)

The loss over the epochs is shown.  While the generated cats are noticeable better than before, they are still not what someone would probably actually draw.

![simple_loss](https://user-images.githubusercontent.com/14242505/50039098-1b90d080-ffe1-11e8-971e-c731a34dd427.png)


# DCGAN

The DCGAN performs much better.  DCGAN.ipynb contains a more sophisticated implementation of a GAN.  The class of doodles we chose to use are also cats for comparison reasons.  The architecture is similar with convolutional layers and batch normalization layers added in.  The architecture is shown below.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_66 (Conv2D)           (None, 14, 14, 64)        1664      
_________________________________________________________________
leaky_re_lu_59 (LeakyReLU)   (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_37 (Dropout)         (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_67 (Conv2D)           (None, 7, 7, 128)         204928    
_________________________________________________________________
leaky_re_lu_60 (LeakyReLU)   (None, 7, 7, 128)         0         
_________________________________________________________________
dropout_38 (Dropout)         (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_19 (Flatten)         (None, 6272)              0         
_________________________________________________________________
dense_32 (Dense)             (None, 1)                 6273      
=================================================================
Total params: 212,865
Trainable params: 212,865
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_33 (Dense)             (None, 6272)              633472    
_________________________________________________________________
reshape_14 (Reshape)         (None, 7, 7, 128)         0         
_________________________________________________________________
batch_normalization_26 (Batc (None, 7, 7, 128)         512       
_________________________________________________________________
up_sampling2d_22 (UpSampling (None, 14, 14, 128)       0         
_________________________________________________________________
conv2d_68 (Conv2D)           (None, 14, 14, 128)       147584    
_________________________________________________________________
leaky_re_lu_61 (LeakyReLU)   (None, 14, 14, 128)       0         
_________________________________________________________________
batch_normalization_27 (Batc (None, 14, 14, 128)       512       
_________________________________________________________________
up_sampling2d_23 (UpSampling (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_69 (Conv2D)           (None, 28, 28, 64)        73792     
_________________________________________________________________
leaky_re_lu_62 (LeakyReLU)   (None, 28, 28, 64)        0         
_________________________________________________________________
batch_normalization_28 (Batc (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_70 (Conv2D)           (None, 28, 28, 1)         577       
=================================================================
Total params: 856,705
Trainable params: 856,065
Non-trainable params: 640
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_41 (InputLayer)        (None, 100)               0         
_________________________________________________________________
Generator (Model)            (None, 28, 28, 1)         856705    
_________________________________________________________________
Discriminator (Model)        (None, 1)                 212865    
=================================================================
Total params: 1,069,570
Trainable params: 856,065
Non-trainable params: 213,505
_________________________________________________________________
```

At first, the generated cat drawings don't look very good.

![bad_conv](https://user-images.githubusercontent.com/14242505/50039234-454af700-ffe3-11e8-9d6d-0e258adc18d3.png)

After training many epochs, it gets considerably better!  

![good_conv](https://user-images.githubusercontent.com/14242505/50039235-49771480-ffe3-11e8-8cdd-8893c7839946.png)

The loss over epochs is shown. The generator loss seems to be diverging, however we were visually inspecting the generated samples on each epoch and the results were improving. 

![conv_loss](https://user-images.githubusercontent.com/14242505/50039236-4d0a9b80-ffe3-11e8-8176-c27411a68c92.png)

# Optimizations

Gaussian noise was added to the images batches for the discriminator to help stability.  A technical discussion on instance noise can be found here: https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/


Many other optimizations were implemented as well.  Specifically, optimizations found here (https://github.com/soumith/ganhacks) are used.  The relevant sections are copied from their README:

#### 1. Normalize the inputs

- normalize the images between -1 and 1
- Tanh as the last layer of the generator output

#### 3: Use a spherical Z
- Dont sample from a Uniform distribution
- Sample from a gaussian distribution
- When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B
- Tom White's [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) ref code https://github.com/dribnet/plat has more details


#### 4: BatchNorm

- Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
- when batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).

#### 5: Avoid Sparse Gradients: ReLU, MaxPool
- the stability of the GAN game suffers if you have sparse gradients
- LeakyReLU = good (in both G and D)
- For Downsampling, use: Average Pooling, Conv2d + stride
- For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
  - PixelShuffle: https://arxiv.org/abs/1609.05158

#### 6: Use Soft and Noisy Labels

- Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
  - Salimans et. al. 2016
- make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator

#### 9: Use the ADAM Optimizer

- optim.Adam rules!
  - See Radford et. al. 2015
- Use SGD for discriminator and ADAM for generator

#### 10: Track failures early

- D loss goes to 0: failure mode
- check norms of gradients: if they are over 100 things are screwing up
- when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
- if loss of generator steadily decreases, then it's fooling D with garbage (says martin)

#### 13: Add noise to inputs, decay over time

- Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
  - Improved GANs: OpenAI code also has it (commented out)



# Classification

We also trained a classifier on 10 classes of doodles.  The notebook can be found at CNNclassifier.ipynb.  After only 3 epochs, it was able to acheive an accuracy of 0.7972!

```
Epoch 1/3
238515/238515 [==============================] - 106s 443us/step - loss: 1.2028 - acc: 0.6102
Epoch 2/3
238515/238515 [==============================] - 105s 442us/step - loss: 0.7889 - acc: 0.7571
Epoch 3/3
238515/238515 [==============================] - 105s 439us/step - loss: 0.6958 - acc: 0.7972
```

If we had more time, we would experiment with adding the generated cat samples into the training data for the CNN classifier.  It would be interesting to see if GANs can be used as a way to augment data and help classification.  This can be especially useful when the number of examples for a certain class is very limited.  By learning how to generate new examples from this under-represented class, a classifier might be able to learn how to recognize that class better.
