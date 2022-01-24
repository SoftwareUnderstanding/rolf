# CIFAR10-CNN-by-Keras
In this project, we investigated the performance of different convolutional network architectures on the CIFAR10 data. We implement from a simple CNN model, then adding layers, doing data augmentation and try wide residual network, and build our models on the CIFAR10 data and learn how different parameters can affect the performance. Above all, we tried 11 varying models and got testing accuracy ranging from 66.20% to 89.38%.
## Project goals
1.	Learning how to process the image data.
2.	Research on different models and build our own model by using Keras.
3.	Compare different architectures and give an explain of the results.
## CIFAR10 description
I will describe the layout of the dataset. The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.
The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch.
For each batch files:
	data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 color image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
	labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data. The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
	label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
## Data processing
We need to convert the labels in the dataset into categorical matrix structure from 1-dim numpy array structure. Then normalize the images in the dataset.
I split the CIFAR10 dataset (60000 images) into training (50000 images) and test (10000 images) datasets to evaluate the model.
Data preparation is required when working with neural network and deep learning models. The image augmentation API is simple and powerful. We will use the Keras Image Augmentation API to finish our data processing. Keras provides the ImageDataGenerator class that defines the configuration for image data preparation and augmentation. We mainly use the capabilities as follow:
	rotation_range: Int. Degree range for random rotations.
	width_shift_range: Float (fraction of total width). Range for random horizontal shifts.
	height_shift_range: Float (fraction of total height). Range for random vertical shifts.
	horizontal_flip: Boolean. Randomly flip inputs horizontally
## Model 1 Simple model
To start with, we build a simple CNN classifier from Keras and explore details to understand how convolution works. Comparing the different batch size (32, 64, 128) and choosing one batch size number for the following models.
We trained the model with 100 epochs.
## Model 2: More-layers model
We saw previously that shallow architecture was able to achieve 68.34% accuracy only. So, the idea here is to build a deep neural architecture as opposed to shallow architecture which was not able to learn features of objects accurately.
We build Model 2-1 base on reference [4] Model 2-2 add layers base on Model 2-1, Model 2-3 add layers base on Model 2-2. We want to compare the test accuracy of the three models and display the results.
We will build the convolutional neural network with batch size 128 and 200 epochs. the network model will output the possibilities of 10 different categories (num_classes) can belong to the image.
## Model 3: Data augmentation
We can increase the number of images in the dataset with the help of a method in Keras library named “ImageDataGenerator” by augmenting the images with horizontal/vertical flipping, rescaling, rotating and whitening etc. 
We got this idea of data preprocessing from reference [5]. First, set up the deep network architecture. Then doing data augmentation and regularizatiojn.
## Model 4: Wide Residual Network
Deep residual networks were shown to be able to scale up to thousands of layers and
still have improving performance but very slow to train[8]. The wide residual networks (WRNs) with only 16 layers can significantly outperform 1000-layer deep networks Also, wide residual networks are several times faster to train[9]. 
We want to implementation of Wide Residual Networks from the paper Wide Residual Networks in Keras. According to the paper, one should be able to achieve accuracy of 96% for CIFAR10 data set[7].
The WRN-16-8 model has been tested on the CIFAR 10 dataset. It achieves a score of 86.17% after 100 epochs. Training was done by using the Adam optimizer. 
## Reference
[1] Plotka, S. (2018). Cifar-10 Classification using Keras Tutorial - PLON. [online] PLON. Available at: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/ [Accessed 16 Dec. 2018].

[2] Keras.io. (2018). About Keras models - Keras Documentation. [online] Available at: https://keras.io/models/about-keras-models/ [Accessed 16 Dec. 2018].

[3] Moncada, S. and Moncada, S. (2018). Convolutional Neural Networks (CNN): Step 2 - Max Pooling. [online] SuperDataScience - Big Data | Analytics Careers | Mentors | Success. Available at: https://www.superdatascience.com/convolutional-neural-networks-cnn-step-2-max-pooling/ [Accessed 16 Dec. 2018].

[4] Kingma, D. and Ba, J. (2018). Adam: A Method for Stochastic Optimization. [online] Arxiv.org. Available at: https://arxiv.org/abs/1412.6980 [Accessed 16 Dec. 2018].

[5] Medium. (2018). [Deep Learning Lab] Episode-2: CIFAR-10 – Deep Learning Turkey – Medium. [online] Available at: https://medium.com/deep-learning-turkey/deep-learning-lab-episode-2-cifar-10-631aea84f11e [Accessed 16 Dec. 2018].

[6] Kumar, V. (2018). Achieving 90% accuracy in Object Recognition Task on CIFAR-10 Dataset with Keras: Convolutional Neural Networks. [online] Machine Learning in Action. Available at: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/ [Accessed 16 Dec. 2018].

[7] Ioffe, S. and Szegedy, C. (2018). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [online] Arxiv.org. Available at: https://arxiv.org/abs/1502.03167 [Accessed 16 Dec. 2018].

[8] GitHub. (2018). szagoruyko/wide-residual-networks. [online] Available at: https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch [Accessed 16 Dec. 2018].

[9] GitHub. (2018). titu1994/Wide-Residual-Networks. [online] Available at: https://github.com/titu1994/Wide-Residual-Networks [Accessed 16 Dec. 2018].

[10] Zagoruyko, S. and Komodakis, N. (2016). Wide Residual Networks. [online] Arxiv.org. Available at: https://arxiv.org/pdf/1605.07146.pdf [Accessed 16 Dec. 2018].
