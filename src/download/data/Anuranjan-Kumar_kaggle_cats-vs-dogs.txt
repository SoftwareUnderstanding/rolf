# Repo for Cats-Vs-Dogs challenge on Kaggle 
The training archive contains 25,000 images of dogs and cats. The size of dataset (Training and Test) is 814.4 MB. For more details, please check https://www.kaggle.com/c/dogs-vs-cats/overview

**1. Dog-vs-Cats.ipynb: Accuracy of 90.6% acheived on model trained from scratch.**
**2. Dogs vs. Cats (VGG16 Fine Tuning).ipynb: Accuracy of 96.5% acheived on model trained on VGG16 with fine tuning.**

There are many pre-trained models available for classification task like VGG16/19, ResNet50, InceptionV3 etc. The reasons for choosing VGG16 were: 
  a) Architecture is fairly simple giving high interpretability.
  b) Easy to visualize the newtork
  c) High accuracy

Some details for VGG16 network (https://arxiv.org/pdf/1409.1556.pdf) are as follows. 
  1. Use of very small convolutional filters, e.g. 3×3 and 1×1 with a stride of one.
  2. Use of max pooling with a size of 2×2 and a stride of the same dimensions.
  3. The importance of stacking convolutional layers together before using a pooling layer to define a block.
  4. Dramatic repetition of the convolutional-pooling block pattern.
  5. Development of very deep (16 and 19 layer) models.
  
  ![vgg1](./images/vgg.png "VGG16 Architecture") ![vgg2](./images/imagenet_vggnet_table1.png "VGG16 Architecture")


#### References:
1. https://arxiv.org/pdf/1409.1556.pdf
2. https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/
3. https://keras.io/api/applications/#vgg16
