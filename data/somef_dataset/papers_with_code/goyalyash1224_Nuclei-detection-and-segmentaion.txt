# Nuclei-detection-and-segmentaion
In computer-aided image analysis of histopathology could provide support for early detection and imporved characterization of different diseases. Nucleus detection and segmentation is a challenging problem due to complex nature of histopathology images. 

We took the dataset from kaggle competetion. The link for dataset is:
            https://www.kaggle.com/c/data-science-bowl-2018/data
We referenced 'U-Net: Convolutional Networks for Biomedical Image Segmentation' research paper which can be found here:
           https://arxiv.org/abs/1505.04597
           
           
The goal for the project is to find the correct number and location of all  nucei shown in the test images. Here in the first stage
training and test data set consisting of 670 and 65 microscopic images of varying size showing ensembles of cells and their nuclei.
The performance of the algorithm is evaluated on the mean average precision at different intersection over union (IoU) thresholds.

We implement a deep neural network of the U-net type consisting several convolutional and max-pooling layers.Prior to training the network we resize, normalize and transform the images. We use 10% of the training data for validation. Furthermore, we implement data augmentation by making use of translations, rotations, horizontal/vertical flipping and zoom. 
