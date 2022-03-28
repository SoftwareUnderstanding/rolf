# A Ship or an Iceberg?
### Project Description
The aim of this project is to buid a classifier that can identify if a remotely sensed target is a Ship or a drifting Iceberg. This is an attempt to solve the problem stated in one of the competitions at Kaggle.com ([here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)).
### Dataset
The dataset for this project is borrowed from its Kaggle competition page (link to dataset [here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data))

The data is provided in .json format (train.json and test.json). The files consist of a list of images and for each image the following fields
* **id**: id of the image
* **band_2, band_2**: flattened image data. each band list consist of 5627 elements corresponding to 75x75 pixel values. (true meaning of these values still need to be understood [[satellite imagery](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background)]).
* **inc_angle**: inclination angle at which the image was taken.
* **is_iceberg**: this field exists only in train.json which if set to 1 indicates that the image is an iceberg, and 0 if its is a ship.

### Milestones
In order to work for this project, Some milestones have been agreed upon to mark the progress of the project.
- **July 13th 2018** By this date we will be able to complete the background needed for this project and will be able to come up with a simple CNN network in python tensorflow.
- **July 25th 2018** By this date we will be able to finalize the simple CNN classifier that we will create, by finalize I mean applying some techniques to increase the accuracy. Moreover, we will be able to come up with a pre-trained classifier (possibly VGG network) trained on the training set for this problem.
- **August 2nd 2018** The finilaization of both the processes (Simple Classifier and pre-trained network) will be done and we will be ready to present our work.

### Work Log
* **july 6th 2018** **Shakti and Nikhil** working on the backgroud of Satellite Imagery
  * Synthetic-aperture Radar (SAR) [Wiwkipedia link](https://en.wikipedia.org/wiki/Synthetic-aperture_radar)
  * Deep Learning for Target Classification from SAR Imagery [link to paper](https://arxiv.org/pdf/1708.07920.pdf)
  
* **july 13th 2018** A simple convolution neural network is has been created. We are using 3 convolution layers and a fully connected layer to get predictions. the details for network are listed below :
  * Input : flattened data points (shape = batch_size x 5625) of 75 x 75 images 
  * Output : one-hot vector of predicted class (shape = batch_size x 2) 
  Please refer file CNN.py for the code.
  * Convolution layers :

    | Layer Index |   inputs    |   outputs   | filter shape | stride | pooling-stride | activation | 
    | ----------- |:-----------:| -----------:| ------------:| ------:| --------------:| ----------:|
    |      1      | -1x75x75x1  | -1x38x38x32 |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      2      | -1x38x38x32 | -1x19x19x64 |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      3      | -1x19x19x64 | -1x10x10x128|     5x5      |   1    | max pooling - 2|   ReLU     |
  * Fully connected layer :
    * number of nodes : 1024
    * input : 10 x 10 x 128 (reshaped to ? X 1024)
    * output size : batch_size x 1024
    * activation : ReLU
  * An attempt to apply dropout is being done (more work on it comming soon)
* **july 14th 2018** A simple 3D convolution neural network is been created. We are using 1 convolution layers and a fully connected layer to get predictions. the details for input and output for network are listed below :
  * Input : flattened data points (shape = batch_size x 16875) of 75 x 75 x 3 images 
  * Output : one-hot vector of predicted class (shape = batch_size x 2) 
  Please refer file CNN3D.py for the code.
* **july 19th 2018** Further detailed exploration of the dataset is done by visualizing random samples from the dataset for each class (file : data_exploration.ipynb). What we found out is that band_2 of most of the images given is comparatively noisy and apparently is not helping enough. We tried to create another cahnnel by combining the given two channels(sum or average of band_1 and band_2) and found out that even if we sum the two given bands, we get a channel that is less less noisy and can be used as a third channel in our convolution model. Although there wasn't any significant difference between the two combinations (sum and average), I personally found the summed version more helpful(we'll see how will that work out in the model).
  * The model that won the original competition isn't available publically, but other baseline models having fairly high accuracy are available publically. Most of them are implemented in keras (actually we didn't find any implemented in TensorFlow), so we are trying to get it to run as soon as possible.
  * Work is also in progress for our version of convolution network in TensorFlow. We will be modifying our 2d convolution implementation to work with three channels (band_1, band_2, band_1+band_2) and training stats will be shared here soon.
  * One more thing that I am working on is to add more images to the trainig dataset. I have gone through several data augmentation techniques that we can apply to availabel images in order to generate new images. these techniques include :
     * Rotation of image around its centre
     * zoomed crop of an image (since the objects are centered in all the images). may be take a 50x50 or 60x60 center crop of some images
  
  It would be helpful to increase the dataset to better train the network.

* **july 20th 2018** In order to increase our samples we tried data augmentation on our training images. We tried flipping it horizontally, vertically, rotating it 90 degrees and shifting the image and created more data in order to increase size of our training dataset. The code for data augmentation is given in DataAugmentation.py file. We have also forked one keras submission of CNN for the same project and tried running it in order to compare the performance of our model with it. 

* **july 25th 2018** The tensorflow CNN implementation that we were working on is finally giving us some acceptable accuracy that we can look upon to. The configutation of the model is shared below.
  * Convolution layers followed by max-pooling:

    | Layer Index |   inputs     |   outputs    | filter shape | stride | pooling-stride | activation | 
    | ----------- |:-----------: | -----------: | ------------:| ------:| --------------:| ----------:|
    |      1      | -1x75x75x1   | -1x38x38x64  |     3x3      |   1    | max pooling - 2|   ReLU     |
    |      2      | -1x38x38x64  | -1x19x19x128 |     3x3      |   1    | max pooling - 2|   ReLU     |
    |      3      | -1x19x19x128 | -1x10x10x256 |     3x3      |   1    | max pooling - 2|   ReLU     |
  * Fully connected layer :
    * number of nodes : 512
    * input : 10 x 10 x 256 (reshaped to ? X 512)
    * output size : batch_size x 512
    * activation : ReLU
  * output layer :
    * 512 inputs and 2 outputs (class probabilities)
  * The above mentioned model was run for 3 epochs with a batch size of 2 and the log loss for has come down to ~2.0. The labelled trainig dataset having ~1600 images was divided inti test and train datasets (keeping 90% for training and 10% for testing). we were able to achieve ~75% accuracy on this model. Keeping in mind that no data augmentation technique is applied to this point, this accuracy seems fair.
  * Although we managed to get this accuracy, the loss was going up and down throughout the trainig process (may be a case of oscillations), we are currently looking into it and will be updating insights soon.
  * work is in progress for applying regularization (L2) on this model and trainng on augmented dataset will be carried on soon and the results will be reported soon.
* An extensive exploration of the images in the dataset is being done in data_exploartion.ipynb notebook. through visualizing some random samples of both classes, we were trying to find out that if the two classes are visually separable. we found out that although some images are vuisually seperable, we cannot generalize anything. Moreover, the band_2 is very noisy in a lot of cases, we came up with a third channel (band_1 + band_2) that has proved to be very effective in eliminating the noise.(visualized in the notebook)
* I (Shakti) am also working on trainig a pretrained VGG model (details will be updated soon)
* For next week, I will be working on VGG model and making changes to out CNN implementation, while Nikhil will be working on Feature extraction and Baseline model on Keras.

* Details on pertrained VGG network : Since last week (week of 4th August), we were working on training a pre-trained VGG16 network, which we have accomplished successfully. The pretained model that is uded is built in keras with tensorflow as background. It is performing pretty well on the data (~82% accurate after training for 10 epochs with a batch size of 50 images.)
* Since the VGG16 architecture have 16 convolution layers and the pretrained weights are of trainig of network on real world images(animals, faces, cars etc) and we are dealing with low resolution satellite images, we decided to not use the later convolution layers(after #5). So, we are using only initial 5 conviolution layers of pretarained VGG16 and then added two fully connected layers to it(relu activation).
* We came to this above conclusion after reading at several sources on why and how to finetune pretrained networks(links in refrences).
* Data augmentation is also applied before feeding the data to pretrained network (keras makes it so easy)
* we set the pretained layers as non trainable and only trained the new dense layers that we added.
* we are now experimenting with more layers on top of those initial five layers, the strategy is as follows:
  * use 16 layers and newly added dense layers, set the 16 convolution layers as non-trainable and perform training on only the dense layers
  * in second phase, set some later convolution layers trainable while still keeping 5 initial layers non-trainable.
* Also, a minor rise in accuracy is experienced while training our implementation of CNN on dataset created by data augmentation.
### References
- Background on Satellite imaging https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background
- Inspiration of Data Augmentation is taken from https://machinelearningmastery.com/image-augmentation-deep-learning-keras/ but we implemented it using numpy methods unlike keras implementation mentioned in the above link.

- Readings on why and How to finetune pretrained networks 
  - https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html
  - https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.htm
- Further reading on finetuning :
  - https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs
  - http://cv-tricks.com/keras/fine-tuning-tensorflow/
- VGG paper https://arxiv.org/pdf/1409.1556v6.pdf
- paper on Salinecy maps https://arxiv.org/pdf/1610.02391v1.pdf

# code files
## resnet.py, Resnet.ipynb--RESNET Implementation
## MultiLayerPerceptron.py-- MLP Implementation
## KerasBaselineImplementation.py, KerasBaseline.ipynb-- Keras Baseline model
