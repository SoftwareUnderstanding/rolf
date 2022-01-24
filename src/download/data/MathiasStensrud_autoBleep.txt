# autoBleep
#### Mathias Stensrud
A neural net enabled video project to read lips, anticipate bad words and put a stop to all that rudeness.

## The Project
The AutoBleep is a program designed to bleep live video/audio, using video trained on a subjects face. The objective is for the program to detect the first part of a bad word though live video analysis, then play a loud sound before the word is finished, effectivly censoring the curse word in real time.

## The Tech
The project is built off of openCV and keras primarily. An openCV program running a haar classification cascade grabs pictures of a subject's mouth every few frames, then runs them through the model. The model itself uses the VGG16 neural net model, but runs every image in a sequence through the model before sending all of these weights into a bidirectional LSTM predictor. This results in a much higher accuracy and better understanding of time in the model.

## The Programs
I wrote an assistant program to generate my dataset, which prompts a user with a word for them to pronounce. Once they say the word and press the key to continue, the past 15 frames are recorded, their mouth is found in the image sequence, and a cropped image is saved to the data folder for their respective status as a word to be censored or not. This is achieved through skimage and haar classifers, to get the image to 140x140 and saved.


## The Terminology
### Haar Cascades
Haar Cascades are sets of Haar-like features, which are very weak classifiers individually, but which can compute at incredible speed. These are sets of rectangles of various intensities that are compared together to predict features.
Haar cascades were used in the first real time face detector, opeing up a new area of possibilites in computer vison and image classification. The main benefit is their very quick classification speed compared to almost any other image classifier. They have been integrated with OpenCV to a great deal, aiding in their use.

### Recurrent Neural Networks and LSTM Layers
God, this is the _fun_ stuff. Neural nets are seen as a bit of a 'black box' by a lot of people, and that adds to their allure. While theyr are specially equipped to deal with certain problems, they are most definetly not an overall solution.
One problem that they excel at is image classification, though that is traditionally covered by Convolutional Neural Nets.
Recurrent Neural Nets, however, are able to maintain a longer memory than CNNs, making them useful for sequence data, such as that from a video. LSTM networks have a very long memory which really helped boost the efficacy of my model.

### Neural Nets and Transfer Learning
Training an accurate neural net generally takes a great deal of time, and also requires a large amount of training data. That is why many people use pretrained models, which they reuse the weights of, in their own projects.

## Results
### The Lows
My model had an accuracy less than ideal. This is most likely due to a paucity of datapoints. I recorded myself speaking ~750 words, acoounting over 4000 images in total, but as they were in sequences, I only had those ~750 data points. Comparatively, a paper I read on a similar topic had 1500 datapoints, from 10 volunteers.
This isn't the end of the world, however, as my program is designed to work modularly, and the model can be swapped out for a better trained one with ease.
### The Highs
The main prompt script I wrote ended up being the most helpful part of the project, and something that I can see myself using in the future, to further improve this and other projects.


## Citations
http://cs231n.stanford.edu/reports/2017/pdfs/227.pdf -Inspiration and model design help
https://arxiv.org/abs/1409.1556 -VGG16 Paper Citation
