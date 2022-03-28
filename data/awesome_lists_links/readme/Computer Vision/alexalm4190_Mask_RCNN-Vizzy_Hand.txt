# Mask_RCNN-Vizzy_Hand

The purpose of this repository is to use [Mask RCNN](https://arxiv.org/abs/1703.06870) for 2D segmentation of a humanoid 
robotic hand. 

For the implementation of the network's architecture, we use the [matterport implementation](https://github.com/matterport/Mask_RCNN).
This library can be found in [my_mrcnn](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/my_mrcnn).
The reason we include the library in our repository is due to some minor changes we did in their code.

In [hand](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/hand) is where we have our main code to handle the 
datasets, extend some functions of [my_mrcnn](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/my_mrcnn), configure 
hyperparameters and create the training/inference processes. 

In [utils](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/utils) we also provide some utility functions, like
evaluation metrics, pre-processing functions, an annotation tool to generate groundtruth masks from real images, amongst other 
utilities.

To generate images for training and validation, we also provide a Unity framework to generate simulated images, available in 
[Unity_package](https://github.com/alexalm4190/Mask_RCNN-Vizzy_Hand/tree/master/Unity_package). 


### Training

- Prepare train/val data: Place the RGB images into a folder called "images" and the groundtruth binary masks into a folder called
"mask". Both folder must be in the same directory. The masks are RGB images, where the positive pixels have a RGB value of 
(0, 0, 0) and negative pixels have a RGB value of (255, 255, 255).

- Download [Mask RCNN COCO pre-trained weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).

- To train the network, for the new task, run the terminal command (inside the "hand" folder): 
python3 hand.py -m=train -d=/path/to/logs -i=/path/to/train_val_images_masks -w=/path/to/pre-trained_weights


### Inference

- To evaluate a model, on the validation set used to configure hyperparameters, run the terminal command:
python3 hand.py -m=test -d=/ -i=/path/to/train_val_images_masks -w=/ -p=/path/to/model

- To get the evaluation on a new test dataset, unseen by the model, run the terminal command:
python3 hand.py -m=test -d=/ -i=/ -w=/ -t -p=/path/to/model




## Results of our final model on test images

<img src="https://cdn.discordapp.com/attachments/351406198874177537/643104048483926029/0.png" alt="Test image 1" width=500/>

<img src="https://cdn.discordapp.com/attachments/351406198874177537/643104110031142912/2.png" alt="Test image 2" width=500/>

<img src="https://cdn.discordapp.com/attachments/351406198874177537/643104155308916781/8.png" alt="Test image 3" width=500/>

<img src="https://cdn.discordapp.com/attachments/351406198874177537/643104244328824852/14.png" alt="Test image 4" width=500/>
