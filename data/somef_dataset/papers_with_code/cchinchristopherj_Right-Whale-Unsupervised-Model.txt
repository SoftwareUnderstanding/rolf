Right Whale Detection Challenge Part II
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls (Unsupervised Learning of Filters)

Tackles the same [challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) and uses the same [dataset](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) as the Baseline Model in [Right Whale Detection Challenge Part I](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network)

Due to expert annotation being an expensive process, unsupervised learning holds great potential for the development of more robust classification models - its speed (in the case of K-Means-like algorithms) and ability to learn abstract, generalized features from readily available unlabeled data make it a powerful tool for future monitoring systems. 

In order to potentially improve the performance of the baseline (supervised) CNN model, the weights of the convolutional filters will be learned unsupervised instead of through standard backpropagation and gradient descent. Concretely, K-Means Clustering, an approach that clusters data into groups, the members of which bear a strong similarity with one another based on a specified distance metric, will be used to identify optimum filters. 

The rationale for K-Means Clustering being a viable alternative for feature learning can be traced back to the basic operation of a convolutional filter. The filters below, for example, are useful for edge detection:

![edge_detector](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/edge_detector.png)

*Vertical and Horizontal Edge Detector Convolutional Filters. Image Source: [Optimization of Processor Architecture for Image Edge Detection Filter](https://www.semanticscholar.org/paper/Optimization-of-Processor-Architecture-for-Image-Osman-Hussin/4c694c500c1394918e437d9259fd10d8df3d9372)*

Recall that a convolution operation involves computing "matrix dot products" (elementwise multiplications and summations) between patches of an input feature map (or feature map volume) and a learned convolutional filter. If a 3x3 learned convolutional filter contains large magnitude values in the top and bottom row and 0s in the middle row, it will be able to "detect" horizontal edges (since large magnitude values would result from the elementwise multiplication between the top and/or bottom row of the filter and horizontal edges in the input image patches). Essentially, therefore, convolutional filters in fact perform cross-correlation, measuring the similarity between two matrices A and B as a function of the displacement of A relative to B. (It can be mathematically proven that the only difference between convolution and cross-correlation is the “flipping” of B relative to A in the relevant equations). 

Centroids learned by K-Means Clustering can therefore be interpreted as characteristic sets of features of samples in the dataset. 

![kmeans](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/kmeans.png)

*K-Means Clustering Algorithm Identifies K Centroids Corresponding to Clusters in the Dataset. Image Source: [K Means Clustering: Identifying F.R.I.E.N.D.S in the World of Strangers](https://towardsdatascience.com/k-means-clustering-identifying-f-r-i-e-n-d-s-in-the-world-of-strangers-695537505d)*

Reinterpreted as convolutional filters, these centroids can be cross-correlated (or convolved) with input images to determine whether or not they contain these characteristic features, with the output of the convolution operation being "match scores" indicating the degree of similarity between the patches in the input images and the representative centroids. A neural network can use these match scores to discriminate between the classes of interest.

There is, however, one large shortcoming to K-Means Clustering: as the dimensionality of the inputs increases, the algorithm requires larger and larger amounts of data to identify representative centroids. One author for example found that using K-Means on 5,000,000 patches of size (64,64), i.e. an input dimensionality of 64x64=4096, yielded sub-optimal results: many clusters were empty or comprised of only a few samples, meaning that algorithm failed to capture centroids representative of features in the dataset. The author found empirically that, given the same number of patches, performance of the algorithm improved when the patches’ input dimensionalities were reduced by an order of magnitude to the hundreds (such as sizes of (6x6) or (8x8)) instead of the thousands. 

Learning filters for the first convolutional layer of a CNN via K-Means should not be problematic since patch sizes are conventionally chosen to be relatively small (in the range of (3,3) to (11,11)) and the number of channels conventionally ranges from 1 (greyscale) to 3 (RGB color). However, due to the conventional increase in number of channels with depth in a CNN, learning filters for deeper convolutional layers becomes problematic due to the increasing dimensionality of inputs given to the K-Means algorithm. 

In this application, two different approaches for dimensionality reduction will be taken to ensure effective filter learning for the second convolutional layer: 
- Model 1: Energy-Correlated Receptive Fields
- Model 2: 1x1 Convolution Dimensionality Reduction

Model 1 (Energy-Correlated Receptive Fields Grouping)
=========================

Several authors have handled this shortcoming of K-Means by randomly creating smaller groups of feature maps from the entire feature map volume and passing these groups of reduced dimensionality to the K-Means algorithm to learn representative centroids. 

![k_means_grouping](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/k_means_grouping.png)

*Feature Map Grouping for Filter Learning via K-Means. Image Source: [An Analysis of the Connections Between Layers in Deep Neural Networks](https://arxiv.org/pdf/1306.0152.pdf)*

These groups can be created randomly or through a pairwise similarity metric such as squared-energy correlation. (Squared-energy correlation is used rather than correlation because the inputs are, as per convention, already linearly-uncorrelated via PCA). In this approach, a similarity matrix is created where each (j,k) element represents the squared energy correlation between the j-th and k-th individual feature map in the entire feature map volume. 

Groups of group size G can be created by selecting N random rows (N random feature maps) in the similarity matrix. For each of these rows, one can select the G elements with the highest values (corresponding to the feature maps with the strongest correlation) and thereby form N groups of size G. 

These groups of reduced dimensionality can then be given to the K-Means algorithm to learn representative centroids. 

![receptive_field_group](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/receptive_field_group.jpeg)

*Example of a Group Chosen using the Squared-Energy Correlation Metric, demonstrating the elements' similarity in orientation but difference in phase and frequency. Image Source: [Selecting Receptive Fields in Deep Networks](http://robotics.stanford.edu/~ang/papers/nips11-SelectingReceptiveFields.pdf)*

The authors of the approach noted that applying K-Means to these correlated groups, rather than random groups, resulted in better classifier performance: since feature maps that respond similarly are grouped together, the algorithm is now able to even more finely model the patterns and structures they are responding together to in the images. 

Model 2 (1x1 Convolution Dimensionality Reduction)
=========================

Unlike conventional convolutional filters that have receptive fields covering entire patches in each channel of an input, (1x1) convolutional filters have a receptive field of one pixel in each channel, thereby precluding their ability to learn features in local areas. These filters therefore operate in a fundamentally different way, a way that in fact mirrors the operation of fully-connected layers.

Using 1x1 filters can be thought of as sliding a fully-connected layer with n nodes from left to right and top to bottom across the image, as each operation in each position yields n values (just as to be expected for the output of a fully-connected layer with n nodes). 

![1x1conv](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/1x1conv.png)

*1x1 Convolution with 32 filters can be conceptualized as sliding a fully-connected layer with 32 nodes across the input feature map volume, producing 32 values in each position. Image Source: [Lecture 5: Convolutional Neural Networks](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf)*

Since (1x1) convolutions act like fully-connected layers, they allow a model to learn not only additional function mappings, but additional nonlinear function mappings if nonlinear activation functions are used.

Furthermore, 1x1 convolutions are able to reduce the number of channels in an input, in essence summarizing or pooling information across the channels into one feature map. (1x1) convolutional filters are therefore an extremely effective and parameter-efficient means of reducing input dimensionality, and are used in this model to help K-Means learn effective filters for the second convolutional layer.

![1x1_conv_cost](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/1x1_conv_cost.png)

*By incorporating a 1x1 Convolutional "Bottleneck" Layer to perform dimensionality reduction through channel pooling, the total computational cost of the desired 5x5 filter convolution operation is reduced significantly due to fewer required parameters. Image Source: [DeepLearning series: Convolutional Neural Networks](https://medium.com/machine-learning-bites/deeplearning-series-convolutional-neural-networks-a9c2f2ee1524)*

The final tuned model architecture for Model 2 is as depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model1/cnn_architecture_unsup.png)

Results of Training for Model 1
=========================

Model 1 was trained for 17 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 4 hours on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 17 epochs was found to be 94.14%, while the ROC-AUC score for the test set was found to be 93.13%.

**Test ROC_AUC Score = 0.9313**

ROC Curves for Model 1
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelUnsup1_BP](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model1/ROC_ModelUnsup1_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelUnsup1_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model1/ROC_ModelUnsup1_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelUnsup1_All](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model1/ROC_ModelUnsup1_All.png)

Results of Training for Model 2
=========================

Model 2 was trained for 16 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 50 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 16 epochs was found to be 96.07%, while the ROC-AUC score for the test set was found to be 95.97%.

**Test ROC_AUC Score = 0.9507**

ROC Curves for Model 2
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelUnsup2_BP](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model2/ROC_ModelUnsup2_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelUnsup2_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model2/ROC_ModelUnsup2_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelUnsup2_All](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/Model2/ROC_ModelUnsup2_All.png)

Conclusions
=========================

Model 1 and Model 2 did not achieve higher AUC scores on the test set (**93.13%** and **95.07%** respectively) conpared to the Baseline Model (**98.29%**) as desired. In particular, the complexity of Model1's architecture, resulting higher computational and memory costs, and poorer performance disqualified it from further analysis. Model 2, however, showed promise in the simplicity of its unsupervised approach to learning convolutional filters and, with further hyperparameter tuning, Model 2's performance could be optimized to surpass that of the Baseline Model. At this stage, however, the Baseline Model has the advantage of faster training time, simple (standard) CNN architecture (easy implementation), and higher AUC score performance. 

Appendix: Filter Visualization (0th Layer)
=========================

The filters of the 0th convolutional layer in CNNs (applied to the raw input images) are often "human-interpretable" and have patterns that are easy to correlate with patterns of the input images. Both Model 1 and Model 2 learn the same number of filters (256) in the same manner via K-Means for the 0th layer. Examine a visualization of these filters in the grid below:

![filters_unsup](https://github.com/cchinchristopherj/Right-Whale-Unsupervised-Model/blob/master/Images/filters_unsup.png)

*Note: Many patches appear to have patterns from the higher-intensity, brightly yellow-colored areas of the spectrogram containing a right whale upcall, and are similar in appearance to the filters learned by the [Baseline Model](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network). Note, however, that K-Means learned filters using an equal number of samples from the positive class (right whale upcall) and negative class (ambient noise), resulting in patches representative of both types of images. (Including samples from both classes, as opposed to just including samples from the positive class, was found to boost classifier performance). Therefore, the brightly-colored areas of the filters could be interpreted as either high-intensity regions corresponding to an upcall, or high-intensity regions corresponding to blips of random noise.*

*As a final note, recall that these filters/centroids are learned from PCA-whitened patches of contrast-enhanced spectrogram images, thereby precluding direct comparison between patterns in the spectrogram images and patterns in the filters.* 

Modules and Installation Instructions
=========================

**"Standard" Modules Used (Included in Anaconda Distribution):** numpy, matplotlib, pylab, glob, aifc, os

If necessary, these modules can also be installed via PyPI. For example, for the "numpy" module: 

        pip install numpy

**Additional Modules used:** skimage, sklearn, keras

skimage and sklearn can be installed via PyPI. For example, for the "scikit-image" module:

        pip install scikit-image

For Keras, follow the instructions given in the [documentation](https://keras.io/#installation). Specifically, install the TensorFlow backend and, of the optional dependencies, install HDF5 and h5py if you would like to load and save your Keras models. 

Correct Usage
=========================

My trained CNN model architecture and weights for Model 2 are saved in the "model_v2.h5" file. This file can be loaded using the command:

    loaded_model = load_model('model_v2.h5')  
    
Note: "load_model" is a function from keras.models. 
With this model loaded, you can follow the procedure as described in training_v2.py to predict the label of a new audio file that may or may not contain a right whale upcall. 

Due to the complicated architecture of Model 1, it is not possible to directly load the model using Keras' "load_model" command. Instead, use the "model_v1_load.py" function to first re-create the model architecture, then load in the weights to the appropriate layers: 

    python model_v1_load.py 
    
With this model loaded, you can once again follow the procedure as described in training_v1.py to predict the label of a new audio file.

If you would like to replicate the process I used to train the CNN models, perform the following:
First, download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) to the desired directory on your computer.
Then run either:

    python training_v1.py 
    
or:

    python training_v2.py 
    
This constructs the CNN model architectures of Model 1 or Model 2, respectively, trains the filters unsupervised via K-Means, and trains the weights on the dataset. This trained model can be saved to your computer using the command:

    model.save('model.h5')  

References
=========================

Coates A., Ng A.Y. (2012) [Learning Feature Representations with K-Means.](https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf) In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade. Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg

Coates A., Ng A.Y. [Selecting receptive fields in deep networks.](http://robotics.stanford.edu/~ang/papers/nips11-SelectingReceptiveFields.pdf) In: Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., Weinberger, K. (eds.) Advances in Neural Information Processing Systems 24, pp. 2528–2536. Curran Associates, Inc. (2011)

Salamon J., Bello J.P., Farnsworth A., Robbins M., Keen S., Klinck H., et al. (2016) [Towards the Automatic Classification of Avian Flight Calls for Bioacoustic Monitoring.](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0166866&type=printable) PLoS ONE 11(11): e0166866. doi:10.1371/journal.pone.0166866

Culurciello E., Jin J., Dundar, A., Bates J. [An Analysis of the Connections Between Layers of Deep Neural Networks.](https://arxiv.org/pdf/1306.0152.pdf) *arXiv preprint arXiv:1306.0152*, 2013

Lin M., Chen Q., Yan S. [Network In Network.](https://arxiv.org/pdf/1312.4400.pdf) *arXiv preprint arXiv:1312.4400*, 2014

Knyazev B., Barth E., Martinetz T. [Recursive Autoconvolution for Unsupervised Learning of Convolutional Neural Networks.](https://arxiv.org/pdf/1606.00611.pdf) *arXiv preprint arXiv:1606.00611*, 2017

Cakir E., Parascandalo G., Heittola T., Huttunen H.,Virtanen T. [Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection.](https://arxiv.org/abs/1702.06286) *arXiv preprint arXiv:1702.06286*, 2017
