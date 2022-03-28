# Image Classification Using a Convolutional Neural Network

The CIFAR-10 data set is a labeled subset of the 80 million tiny images data set. CIFAR-10 contains images of ten different classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The data set is split into a training set of 50,000 images (5,000 per class) and a test set of 10,000 images (1000 per class). Each image is a 32 by 32 RGB image of one of the ten classes.

*train_model.py* creates and trains a convolutional neural network to classify the test images. The CNN is based off of the VGG-B network outlined in https://arxiv.org/pdf/1409.1556.pdf. VGG is used to classify very large images (224 by 224) so some modifications were made to better fit the smaller data of CIFAR-10. For instance, instead of using five convolutional blocks as in VGG, I used three. And, instead of using 64 filters in the first layer, I used 32.

The model consists of three convolutional blocks and two dense layers. Each convolutional block has two convolutional layers and a max pooling layer. All three blocks use 3 by 3 filters with padding to avoid dimensionality reduction. Each block also uses 2 by 2 max pooling, reducing the dimensions by half. Each block has two convolutional layers followed by a max pooling layer. Both convolutional layers in the first block have output of depth 32. Convolutional layers in the second block have output of depth 64, and 128 in the third block. 

After all the convolution/pooling blocks, the output is flattened and input into the first dense layer with 128 nodes. The final output is run through the second dense layer using softmax to get the ten categorical probabilities. 

Each convolutional layer and the first dense layer use ReLU as an activation function to inject nonlinearity. Batch normalization is performed after every convolutional layer and the first depth layer to stabilize the learning process. To avoid overfitting the training data, dropout is performed after each convolutional block and the first dense layer. The dropout proportion increases the further you go into the network using values of .1, .2, .3, and .4.

The CNN has the following structure:
* Input Layer -> (28 x 28 x 3)
* Convolutional Block 1 -> (14 x 14 x 32)
  * Convolutional Layer 1.1 with ReLU and Normalization (32 filters)
  * Convolutional Layer 1.2 with ReLU and Normalization (32 filters)
  * Max Pooling Layer
  * 10% Dropout
* Convolutional Block 2 -> (7 x 7 x 64)
  * Convolutional Layer 2.1 with ReLU and Normalization (64 filters)
  * Convolutional Layer 2.2 with ReLU and Normalization (64 filters)
  * Max Pooling Layer
  * 20% Dropout
* Convolutional Block 3 -> (4 x 4 x 128)
  * Convolutional Layer 3.1 with ReLU and Normalization (128 filters)
  * Convolutional Layer 3.2 with ReLU and Normalization (128 filters)
  * Max Pooling Layer
  * 30% Dropout
* Flatten -> (1 x 2048)
* Dense Layer with ReLU and Normalization -> (1 x 128)
* Dense Layer with SoftMax -> (1 x 10)

*load_data.py* has two functions *load_training_data* and *load_test-data* that load the training and test sets from the data folder. If you download this project, the data folder will be empty. That is because I did not want to store ~160MB of data in this repository. To download the data, click the "CIFAR-10 python version" link from https://www.cs.toronto.edu/~kriz/cifar.html. Unpack the tar file and copy the contents to the data folder so that it has the following structure:
* data
  * data_batch_1
  * data_batch_2
  * data_batch_3
  * data_batch_4
  * data_batch_5
  * test_batch

*train_model.py* builds the CNN and trains it on the 50,000 train images. It then stores the model and the weights in the model folder. The hyperparameters can be tuned within the code. Each epoch takes 5-6 minutes on my MacBook with an average performance GPU. *test_model.py* loads the model data and tests it on the 10,000 test images. After just 10 epochs, it can classify the test images with ~80% accuracy. This takes my comptuer about an hour to train. I estimate that the model should reach >85% accuracy after ~100 epochs.
