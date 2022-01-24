# Contrastive-Learning


Here, we test the concept of contrastive learning using SimCLR on CIFAR-10 dataset. All of the data available is not labelled and it is very expensive to annotate data. SimCLR which is an unsupervised machine learning technique learns the representations by identifying the similarities and differentiating between the dissimilarities in the batch of images. At an elementary level using contrastive learning SimCLR maximizes agreement among two representations of images. 

Keras and Tensorflow libraries in python formed the skeleton for implementation of the SimCLR algorithm. The various steps to implement the SimCLR algorithm are as follows: 
1.	Data Augmentation
i.	Random Crop and Resize 
ii.	Color Distortion  
iii.	Gaussian Blur
2.	Neural Network Encoder – ResNet50
3.	Projection Head – MLP 
4.	Contrastive Loss Function

Note: Since, we apply 2 different functions of augmentations on the same image, we take a sample of 10 images, we get the output as 2*10 augmented images in a batch. In this way we form pairs of images and to maximize the number of positive-negative matching, we pair each image with another class image. Unfortunately, while implementing the setup we were a hit by an error related to eager execution of the program. In order to sample and batch, tensorflow has inbuilt functions repeat, shuffle and batch but they are available only when the program has a graph for the flow of control which is not possible in eager execution. For future work there has to be a method to control the switching between the normal tensor implementation and eager execution.
Loss Function: In this study, we used Noise Contrastive Estimator loss. If the images are similar, the function figures out the images pair as correlated and form a positive pair. If the images are dissimilar, the function figures out the image pair as uncorrelated and form a negative pair.
We find distance/similarity between the pair of images using cosine similarity. In this way, we will reduce the distance in positive pair and increase the distance in negative pair. Finally, we have the neural network.
Neural Network Encoder:  We used TensorFlow’s inbuilt method of defining ResNet50 as a base encoder. The augmented CIFAR dataset is passed on to ResNet50 as an input.  The output is a high dimensional vector which we then apply a projection head on a Multilayer perceptron with 2 dense layers. The layers have same number of dimensions and a hidden layer has a non-linear activation function - ReLU. 


Refernces : 

https://arxiv.org/abs/2002.05709

https://sthalles.github.io/simple-self-supervised-learning/
