# ResNeXt50-Implementation-and-Transfer-Learning-with-ResNet50V2

This project includes two main parts: Implementing ResNeXt50 from scratch and doing transfer learning with ResNet50V2 to classify normal vs pneumonia chest Xray images (Keras does not have ResNeXt50 trained model yet).

1) ResNeXt was created by Xie et al. (2016). This architecture was developed based on ResNet architecture, which also uses the idea of residual blocks for maintaining information from previous layers. The main difference between ResNeXt and ResNet is instead of having continual blocks one after the other, 'cardinality', which is the size of transformations , was considered and implemented in the architecture, inspiring from Inception/GoogLenet. Compared to ResNet, ResNeXt has fewer parameters but better performance in Imagenet Challenge (lower top-1 and top-5 errors). 

Reference: Xie, S., Girshick, R., Dollár, P., Tu, Z., &amp; He, K. (2017, April 11). Aggregated Residual Transformations for Deep Neural Networks. arXiv.org. https://arxiv.org/abs/1611.05431. 

2) Chest Xray image dataset was downloaded from Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. This class contains 2 classes: normal chest and pneumonia chest images. We will perform transfer learning, which utilizes the well-known trained models for our specific tasks, using ResNet50V2 to determine whether chest from an Xray image is normal or pneumonia. ResNet50V2 was chosen not only because of being compatible with implementation in part 1 but also because it is a light model (not too many parameters) but has a decent performance (pretty average accuracies) as being reported in https://keras.io/api/applications/.
