# PlantNet_MobileNetV2
Google Colab python notebook with training of a MobileNetV2 neural network on PlantNet Dataset. The goal of the network is to classify 1081 different plant species. MobilenNet takes in 224 x 224 x 3 sized images and outputs classification label. 

There is also a CoreML conversion to put it into iOS devices. Conversion to iOS can be tricky but I'll post some code on that a bit later.

Images from PlantNet

![alt text](https://github.com/isakdiaz/PlantNet_MobileNetV2/blob/main/plantplot1.png)

Dataset Source: https://gitlab.inria.fr/cgarcin/plantnet_dataset
MobileNetV2 Paper: https://arxiv.org/abs/1801.04381

Also checkout my other project deeplearning-swift-ios which has a native IOS App that I made using this model.
