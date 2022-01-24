This is a repository to store files while I work on the Eluvio ML Scene Segmentation Challenge.

The submitted codes can be found in the folder 'modified_final'. Previous submission is in the folder 'final'.

Motivation for using Focal Loss as defined in https://arxiv.org/abs/1708.02002 and https://pypi.org/project/focal-loss/:
On observing the data it is seen that there are close to 8000 true scene boundaries and close to 97000 ordinary shot boundaries. This is an example of an imbalanced data set for a one-shot detector that was being developed. Focal loss has been demonstrated to help in such situations and the code is conveniently developed in a Python package.

UPDATE: 11th March 2021 (submitted via email on 2nd March 2021)

New metrics: Mean Average Precision (mAP) = 0.14, Mean Maximum IoU (miou) = 0.29

Previous metrics: Mean Average Precision (mAP) = 0.081, Mean Maximum IoU (miou) = 0.044

Description update: 11th March 2021

The idea was to use a window of shots to generate feature embeddings for the boundary after the central shot. So for examples shots 1 through 7 (assuming 0 index is first) would be used to learn embeddings to represent the boundary after shot 4. 

The main aspects of this approach are:

1) Since this is a sliding window approach the ends of each movie were padded with shots having 0 (zero padding). The number of shots was flexibly varied and is calculated as int((window-1)/2). It is important to note that the size of the window needs to be an odd integer greater than 1 so that each central shot has the same number of neighbors on either side. 

2) Instead of creating copies of the three features 'cast', 'action' and 'audio', I trained four separate convolutional layers on each of the provided features. The learned embeddings were then concatenated before being fed into the dense part of the network.

3) L1 regularization is used in all layers for weights and biases to encourage sparsity in these parameters. Using a sigmoid function was also seen to improve both learning and the final results respectively. Other regularizations, viz. L2 and L1_L2 were tried but did not improve the learning or results by much.

4) Global Average Pooling was used to retain the average of all activation maps in each convolutional layer. This was done primarily to keep the number of connections required to connect to the dense network down, which also helps in reducing the number of free parameters of the network.

5) The model was trained using 2-fold crossvalidation. In this roughly 60% of the data (37 movies) were used to train classifer1 and 40% of the data (27 images) were used to train classifier2. <b> The classifiers were interchanged when testing the model </b>. 

6) The main reason for using 2-fold crossvalidation was due to hardware limitation. It was seen that the the latter part of the data set (27 images) contained more shots/movie than the earlier 37 images. Ideally I would have liked to perform a 5-fold crossvalidation in which, in iteration 1: datasets 1 through 4 would be used for training and 5 would be used for testing. The order of training and testing data sets would be changed in each iteration, thus allowing the model to predict on each fold while treating it as a test data set. However this proved to be a very memory intensive task and could not be completed even on Google Colab Pro with 27 GB of RAM.

Training the model:
The model was trained using a focal loss with alpha = 9 and gamma = 2.5. These values were used after observing that the complete Movie Scenes Dataset has a 9:1 ratio of negative to positive examples and the same value for gamma was reported in https://arxiv.org/abs/1708.02002 

As it can be seen from the figures below the model did not overfit to either training data set during each fold. No improvement in model performance was observed beyond 50 epochs and hence the training was stopped at 50 epochs.

<p align = "center">
  <img src = https://github.com/Yashgh7076/Eluvio-ML-Scene-Segmentation/blob/main/images/lasso_final1.png/>
</p>

<p align = "center">
  <img src = https://github.com/Yashgh7076/Eluvio-ML-Scene-Segmentation/blob/main/images/lasso_final2.png/>
</p>

Challenges faced and workarounds created:
1) The major limitation of this approach is the use of exhaustive hardware. A laptop having 16 GB of RAM can process a dataset when windows = 5, but typically encounters a MemoryError when a larger window size is chosen. This was remedied by breaking the data down into 5 parts manually and extracting features. The link to the created datasets is provided here: https://drive.google.com/drive/folders/10U2EFCuH1fP5Wc0cf7Abn9c29ppJavEx?usp=sharing

2) Even a single convolutional layer for each feature generates a lot of hyperparameters for the network to learn. Hence L1 regularization needs to be used to encourage sparsity. Dropout can also be explored for this approach which would allow more convolutional layers to be added to the network.

Description for other neural network models attempted:

1) cnn.py -> Create copies of 'cast', 'action', 'audio' to create a feature vector of size (2048, 4) for each pair of shots that define a boundary. Then a Siamese CNN is used to learn features from adjoining shots and classify shot boundary as scene boundary or not.

2) dense.py -> Learn a joint dense network on the same data as 1)

3) siamese.py -> Learn a set of CNN each for 'places','cast','action' and 'audio' and combine the activation maps before transferring to a dense network.

4) siamese_2.py -> Learn 2 CNN's for the shots the define the boundary.

5) siamese_3_class_imbalance.py -> Same as siamese_2.py but use BinaryFocalLoss as loss instead of BinaryCrossEntropy

6) simple_cnn.py -> Learn two CNN's for adjoining shots and combine activation map before feeding into a dense network.

7) time_series.py -> Reformat the data such that an individual data frame is of size (window, 3584) where window can be flexibly chosen. This approach considers a sliding window approach to classification by allowing a CNN to see the neighboring shots of a boundary.
