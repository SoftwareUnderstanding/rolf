# Semantic Segmentation Using a Fully Convolutional Network

This is work that I did as part of a group project (with Lingxi Li, Yejun Li, Jiawen Zeng, and Yunyi Zhang) for CSE 251B (Neural Networks).  I benefited from discussions with my partners, but all code here was either given by the instructor or written by me.  Specifically, I created the model (basic_fcn.py) and wrote the code for training the model and evaluating it on the validation and tests sets (starter.py and util.py), and the instructor gave us all the code for loading the data.


This project has a basic fully convolutional neural network that can be used for semantic segmentation.  It uses a subset of the India Driving Dataset to train, validate, and test the models.


## Files

`basic_fcn.py` contains the class for the fully convolutional network.\
`dataloader.py ` contains code for loading training, validation, and test datasets.\
`get_weights.py` contains code for computing weights for each of the classes.  These weights can be used to train a model using weighted cross entropy loss.\
`latest_model.pt` contains a trained model which can be used to make predictions on test images.\
`starter.py` contains code needed to train a model.  If you would like to run this, you'll need to download (some portion of) the India Driving Dataset and save links to each of the training, validation, and test images to files `train.csv`, `val.csv`, and `test.csv`, respectively, in your working directory.\
`utils.py` contains functions used to compute pixel accuracy and IoU for each category.

## Visual Results

Here are some test images where the model performs well.  Each strip contains the actual image, ground truth labels for that image, and model predictions, in that order.

![test1](https://user-images.githubusercontent.com/77809548/110228807-ea01c980-7eb8-11eb-9ae4-b46b0171bee5.png)

![test2](https://user-images.githubusercontent.com/77809548/110228861-70b6a680-7eb9-11eb-867e-08d333628125.png)

![test3](https://user-images.githubusercontent.com/77809548/110228917-db67e200-7eb9-11eb-80e8-3110dbe007fe.png)

Here are some test images where the model doesn't perform very well.  Each strip contains the actual image, groud truth labels, and model predictions, in that order.

![test7](https://user-images.githubusercontent.com/77809548/110228949-3d284c00-7eba-11eb-8203-758f67b949a4.png)

![test4](https://user-images.githubusercontent.com/77809548/110228978-837dab00-7eba-11eb-9243-8fbd828d0651.png)


Clearly, the model does not do very well on classes like 'people', 'billboard', and 'motorcycle' which only account for a small portion of the pixels in the training dataset.  I'm hoping that using dice loss (or weighted cross entropy) should fix this issue.



## Numerical Results

The trained model, which can be loaded from `latest_model.pt` was evaluated on the test set, and gave a pixel accuracy of 0.8134 as well as the following intersection-over-union (IoU) values on each category\
0. Road - 0.903
1. Drivable fallback - 0.416
2. Sidewalk - 0.101
3. Non-drivable fallback - 0.246
4. Person/animal - 0.151
5. Rider - 0.253
6. Motorcycle - 0.317
7. Bicycle - 0
8. Autorickshaw - 0.395
9. Car - 0.461
10. Truck - 0.238
11. Bus - 0.179
12. Vehicle Fallback - 0.246
13. Curb - 0.426
14. Wall - 0.221
15. Fence - 0.065
16. Guard Rail - 0.137
17. Billboard - 0.132
18. Traffic Sign - 0.021
19. Traffic Light - 0
20. Pole - 0.156
21. Obs-str-bar-fallback - 0.130
22. Building - 0.409
23. Bridge/tunnel - 0.344
24. Vegetation - 0.756
25. Sky - 0.942


## Future Work
I'm currently working on improving the model by improving the architecture, the loss function, and the data augmentation used.  Specifically, I am implementing the [U-Net](https://arxiv.org/abs/1505.04597) architecture, and using either Dice loss or weighted cross entropy loss.  
