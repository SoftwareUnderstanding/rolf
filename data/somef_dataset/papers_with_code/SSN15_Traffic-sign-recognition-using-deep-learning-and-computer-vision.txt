The goals / steps of this project are the following:

1. Load the data set (see below for links to the project data set)
2. Explore, summarize and visualize the data set
3. Design, train and test a model architecture
4. Use the model to make predictions on new images
5. Analyze the softmax probabilities of the new images

**Dataset**

The dataset used for this project is German Traffic Sign dataset provied by Institut für Neuroinformatik
The relevant information about the dataset can be found in the following link:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Overview

**Data Set Summary**

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

**Exploratory visualization of the dataset**

Here is an exploratory visualization of the data set. It pulls in a random set of 12 images and labels them with the correct names in reference with the csv file to their respective id's.

![Image of dataimage](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/dataimage.png)

After this, the dataset structure is analyzed by plotting the occurrence of each image class to get an idea of how the data is distributed. This helps in understanding the data distribution for each class and which class lacks the data.

The data distribution of training data is shown below:

![Image of traindatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/traindatasetcount.png)

The data distribution of test data is shown below:

![Image of testdatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/testdatacount.png)

The data distribution of validation data is shown below:

![Image of validdatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/validdatacount.png)

The next step is to convert the images to grayscale because the excess information only adds extra confusion into the learning process.
The gray scale image is shown below:

![Image of grayimage](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/grayscale.png)


The grayscale image is then normalized as it helps in speeding up the training and performance metrics like resources. 
The normalized images is shown below in comparison with original image
![Image of normalized](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/normalizedimage.png)

As discussed earlier, the data distibution is not uniform. To mitigate this problem, data augmentation technique is used.
Here in this project, data is augmented using some computer vision techniques. The randomized modifications are random alterations such as opencv affine and rotation.

![Image of augment](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/original%26augmented.png)

Now, the data distribution is checked by plotting the the training, test and validation data.
After augmentation, the number of data in each class is uniform.

The data distribution of training data after augmentation is shown below:

![Image of traindatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/traindatacountaugment.png)

The data distribution of test data after augmentation is shown below:

![Image of testdatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/testdatacountaugment.png)

The data distribution of validation data after augmentation is shown below:

![Image of validdatacount](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/data%20visualization/validdatacountaugment.png)

**Model Architecture**

The model used in this project is LeNet.
I referred the foolowing paper for the implementation of LeNet Architetcture.
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
The details about the model architecture is shown in the table below.

|    Layer      | Description   |
| ------------- | ------------- |
| Input         | 32x32x1 grayscale image  |
| Convolution 5x5  | 2x2 stride, valid padding, outputs 28x28x6  |
| RELU  |   |
| Max pooling  | 2x2 stride, outputs 14x14x6  |
| Convolution 5x5  | 2x2 stride, valid padding, outputs 10x10x16  |
| RELU  |   |
| Max pooling  | 2x2 stride, outputs 5x5x16  |
| Convolution 1x1  | 2x2 stride, valid padding, outputs 1x1x412  |
| RELU |  |
| Fully connected  | input 412, output 122  |
| RELU  |  |
| Dropout  | 50% keep  |
| Fully connected	  | input 122, output 84  |
| RELU  |   |
| Dropout	  | 50% keep  |
| Fully connected	  | input 84, output 43  |

The training parameters of the model can be seen in the code.

The final model results were:

* Training set accuracy of 100.0%
* Validation set accuracy of 99.1%
* Test set accuracy of 94.1%

**Prediction**

To check the accuracy of the model, 6 random traffic sign images were chosen and used the saved model weights to classify the images.
The results were 100% accurate and the result is shown below.

![Image of predresult1](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult.png)

![Image of predresult2](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult2.png)

![Image of predresult3](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult3.png)

![Image of predresult4](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult4.png)

![Image of predresult5](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult5.png)

![Image of predresult6](https://github.com/SSN15/Traffic-sign-recognition-using-deep-learning-and-computer-vision/blob/master/prediction%20result/predictionresult6.png)


**Reference Links**

1. https://arxiv.org/pdf/1409.1556.pdf
2. https://arxiv.org/pdf/1409.4842.pdf
3. https://arxiv.org/abs/1812.03079
4. https://arxiv.org/abs/1512.02325
5. https://developer.nvidia.com/cudnn
6. https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9




