Python version: 3.6

--------------------------------------------

Clone the repository and install the dependancies from requirements.txt file. Make sure to place the data in data directory under train and test folder or change the paths in the main.py file. 

Link to VGG-16 pretrained weights file: [https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5]

----------------------------------------
I have used **vgg-16** as base feature extractor, I will implement ResNet50 and 100 in future.

---------------------------------------------
This reposiort contains a simple implementation of Faster RCNN presented by  Ross Girshick [https://arxiv.org/abs/1506.01497]

I have tested this on 3 Classes extracted from Open Images v 4. data set. In order to download and extract data from open images, run the extract_data.py script. By default, it will download data for 3 classes ['Person', 'Mobile phone', 'Car']. You can edit the list to get other classes as well. This will also convert and save the data which is required by the model.

----------------------------------------------------------------

Inspired by this article: [https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a]