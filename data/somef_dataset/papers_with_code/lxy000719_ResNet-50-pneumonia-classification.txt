# ResNet-50-pneumonia-classification

Built ResNet50 model using TensorFlow Keras to do the classification of pneumonia images (whether the X-ray image shows normal or pneumonia)

You can download the dataset from Kaggle
Data source from Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Original ResNet50 paper reference:
https://arxiv.org/abs/1512.03385

**ResNet50 architecture as below:**
![Alt Text](https://github.com/lxy000719/ResNet-50-pneumonia-classification/blob/master/image/resnet_kiank.png)

**Identity block:**
![Alt Text](https://github.com/lxy000719/ResNet-50-pneumonia-classification/blob/master/image/idblock3_kiank.png)

**Convolution block:**
![Alt Text](https://github.com/lxy000719/ResNet-50-pneumonia-classification/blob/master/image/convblock_kiank.png)

The original image data set from Kaggle has 3 folders; train, val and test. However, there are only 8 images in the val folder. So I combined the train and val folder into one folder trainval and split the whole dataset into train and val by 8:2 using sklearn train-test-split.

As I train the model on my own laptop (MacBook Pro 15-inch 2015) using CPU, it took about 10 hours? (As I went to sleep when it was training, so not sure the exact time). 

The model achieved 99.69% accuracy on training set and 76.79% accuracy on validation set.

**Training:**
![Alt Text](https://github.com/lxy000719/ResNet-50-pneumonia-classification/blob/master/image/Screenshot%202020-08-19%20at%202.29.49%20PM.png)

**Validation**
![Alt Text](https://github.com/lxy000719/ResNet-50-pneumonia-classification/blob/master/image/Screenshot%202020-08-19%20at%202.38.44%20PM.png)

We can see that there is overfitting problem on the training data.
To do list:
- Fine tune the parameters
- Retrain th model using Cloud computing
- Try using pretrained model in Keras libray
