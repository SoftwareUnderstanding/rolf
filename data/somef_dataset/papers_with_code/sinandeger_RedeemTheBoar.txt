# RedeemTheBoar
This repository contains the code we used for the 2018 Data Science Bowl competition on kaggle (https://www.kaggle.com/c/data-science-bowl-2018). The competition required the implementation of a model that is capable of identifying a range of cell nuclei across varied conditions. 

keras_implementation is the main part of the code. It contains the image preprocessing, an implementation of a U-Net (as defined in https://arxiv.org/abs/1505.04597) using Keras. The U-Net architecture consists of a contracting path and an expansive path, each made of convolutional blocks. 

As the training data was limited in size, and skewed in content (images for which it was easier to mask the nuclei dominated the sample), data augmentation was an important part of this challenge. input_pipeline contains the data augmentation snippet keras_implementation calls during training.

Below is an example of the data, and the output of our code. The top two panels is what is provided in the training data. The top left panel is the image, and the top right panel is the ground truth masks of the cell nuclei. The bottom two panels are the predictions of the neural network. The metric of accuracy for the competition was intersection over union (IoU), and this figure is an example of a high IoU score case.

![Figure 1.](https://github.com/sinandeger/RedeemTheBoar/blob/master/4e1c889de3764694d0dea41e5682fedb265eaf2cdbe72ff6c1f518747d709464.png)

Team members: Sinan Deger ([@sinandeger](https://github.com/sinandeger)), Donald Lee-Brown ([@dleebrown](https://github.com/dleebrown)), and Nesar Ramachandra ([@nesar](https://github.com/nesar)). 
