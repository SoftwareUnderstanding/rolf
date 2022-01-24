# Spoken-Numeric-Digit-detection
Spoken digit recognition using the Mel-frequency cepstral coefficients (MFCCs) and convolution neural networks (CNN). The model can recognize 0-9 spoken digits from .wav audio files by passing their MFCCs as an image input. My model achieves an accuracy of 94% on the test audio files. 

## Tensorflow speech command dataset

An audio dataset of spoken words (https://arxiv.org/abs/1804.03209) designed to help train and evaluate keyword spotting systems. I have extracted the spoken digits wave files from the dataset that contains 30 different commands. There are 21,114 training and validation audio files. The rest i.e., 2552 are the test files.

## Extracting the MFCCs for the audio files
Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum").
    
After extracting the MFCCs, they are padded to create 32x32 MFCC based images. Then these images are used to train the CNN.

## Convolution neural network

The CNN architecture used in this project has three stages of convolution models followed by dense connected layer. Here is the architecture of the CNN model used:

```INPUT -> [CONV -> RELU -> CONV -> RELU -> CONV -> RELU -> POOL -> DROUPOUT]*[32,64,128] -> [FC -> RELU]*[128,64] -> [FC -> SOFTMAX]```

Inspired from ResNet :  Kaiming He et al. https://arxiv.org/abs/1512.03385

Building the CNN model and training the model with training datatset audio files with the labeled spoken digits. I have used a batch size of 128 and 40 epochs to train the model.
