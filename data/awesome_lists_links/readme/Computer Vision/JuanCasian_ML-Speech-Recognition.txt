# Machine Learning Speech Recognition
This repository includes my work from the kaggle challenge: [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). The objective of this competition is to create a machine learning model which can recognize between 12 different commands:  yes,  no,  up,  down,  left,  right,  on,  off,  stop,  go,  silence and unknown. Inside unknown, you are supposed to include all the commands that don't fit the first 11 classes. 

### Data
The [data](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data) for this challenge includes more than 28,000 wav files, which are divided into 30 different classes. They come in a 7z compressed format, which can be uncompressed downloading 7zip unzipper. Here are the commands to download and unzip files with 7zip:
```
# Installing the unzipper
sudo apt-get update
sudo apt-get install p7zip-full

# unzipping files
7z x <FILE NAME>
```

### Approach
My approach to solving this challenge was to use a Convolutional Neural Network working with the spectrograms of the audios. In order to do this, the first step was to separate the background noise into smaller fragments because I needed more data points of silence; [ffmpeg](https://www.ffmpeg.org/) tool was really useful for this.

After gathering the extra data I divided the files into its classes in different directories and converted the audio files into spectrograms. In order to do so, I installed sox which allowed me to do the conversion from wav files into spectrograms. 

### Architecture
The architecture I chose for this model was based on the paper written by Karen Simonyan and Andrew Zisserman titled: [Very Deep Convolutional Neural Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf).
#### Graph
- Resize the image to [160, 160, 3] (Height, Width, Channels)
- Convolutional layer No. 1 | Kernel = [5, 5, 3,32] | Stride = [1,1,1,1] | Activation function = ReLU
-  Convolutional layer No. 2 | Kernel = [5, 5, 32,32] | Stride = [1,1,1,1] | Activation function = ReLU
- Max pool layer No. 1 | Kernel = [1, 2, 2,1] | Stride = [1, 2, 2,1]
- Dropout layer No. 1

- Convolutional layer No. 3 | Kernel = [5, 5, 32,64] | Stride = [1,1,1,1] | Activation function = ReLU
-  Convolutional layer No. 4 | Kernel = [5, 5, 64,64] | Stride = [1,1,1,1] | Activation function = ReLU
- Max pool layer No. 2 | Kernel = [1, 2, 2,1] | Stride = [1, 2, 2,1]
- Dropout layer No. 2

- Convolutional layer No. 5 | Kernel = [5, 5, 64,64] | Stride = [1,1,1,1] | Activation function = ReLU
-  Convolutional layer No. 6 | Kernel = [5, 5, 64,64] | Stride = [1,1,1,1] | Activation function = ReLU
-  Convolutional layer No. 7 | Kernel = [5, 5, 64,64] | Stride = [1,1,1,1] | Activation function = ReLU
- Max pool layer No. 3 | Kernel = [1, 2, 2,1] | Stride = [1, 2, 2,1]
- Dropout layer No. 3

- Flatten output from the dropout layer
- Fully Connected layer | Nodes = 1024 | Activation function = ReLU
- Fully Connected layer | Nodes = 12 | (Output layer)

### Training
In order to train the model, I used softmax cross-entropy as a loss function, taking the mean of it to optimize the model. The optimizer I chose is Adam optimizer with a learning rate of 0.001 and a batch size of 32 images. In order to train the model, I used 1 GPU instance in GCP, training for 9000 steps. 
#### Accuracy graph
![Accuracy](res/accuracy_graph.png)
#### Loss graph
![Loss](res/loss_graph.png)
### Results
After training the model for 9000 steps I submitted the results of the challenge data and got 71% of total accuracy after predicting 150,000 audios. 
![Kaggle Results](res/results.png)

### References
- https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
- https://www.tensorflow.org/guide/
- https://www.tensorflow.org/guide/summaries_and_tensorboard
- https://arxiv.org/pdf/1409.1556.pdf
- https://www.ffmpeg.org/

### Dependencies
- tensorflow
- pandas
- numpy
- matplotlib
- os
- skimage
- python
- ffmpeg

### Authors
Juan Pedro Casian - Github: [@JuanCasian](https://github.com/JuanCasian) - Email: juanpedrocasian@gmail.com
