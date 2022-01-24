# Flower-image-classifier

This project is part of the AI Programming with Python Nanodegree from [https://www.udacity.com/] to create 
and train a classifier using transfer learning from three different pre trained CNN to predict flowers images using a dataset with 102 classes.

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on the dataset
- Use the trained classifier to predict flower images

# Dataset

A 102 category dataset, consisting of 102 flower categories, flowers commonly occuring in the United Kingdom. 
Each class consists of between 40 and 258 images, follow this link [http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html]
to the dataset repository to get more detail.

# Pre trained CNN Models used:

- AlexNet : https://arxiv.org/abs/1404.5997
- VGG19   : https://arxiv.org/abs/1409.1556
- ResNet  : https://arxiv.org/abs/1608.06993


# Files in the repository

- **get_train_args.py**   : script to get the command line arguments for the train program     
- **predict.py**          : program to predict the flower image    
- **cat_to_name.json**    : json file to map the classes with the flower names 
- **get_predict_args.py** : script to get the command line argumens for the predict program
- **train.py**            : program to train, validate and test the classifier
- **utils.py**            : utility functions for train and predict apps.

# Install
Clone the repository to the local machine

`$ git clone https://github.com/rafaelmata357/Flower-image-classifier.git`

# Running

The **train** app has the following arguments:

     1. Data Folder for train/val/ test images as --data_dir with default value 'flowers'
     2. Save Dir to save the checkpoint as        --save_dir with default value current directory
     3. Learning rate to used for the optimizer   --learning_rate with default value 0.001
     4. Dropout probability                       --drop_p
     5. Hidden Units to used in the classifier    --hidden_units with default value 512 (one hidden layer)
     6. Epochs number of epocs to used trainning  --epochs with default value 5
     7. CNN Model Architecture to used as         --arch with default value 'vgg19'
     8. GPU to specified gpu resources use        --gpu with default value y or n for CPU

Example to train the classifier using the Alexnet model, with a classifier having two hidden layers of 512 and 256 outputs,
a learning rate of 0.001, 20% dropout probability, 30 epochs and GPU usage, execute:

```$ python train.py --data_dir flower --save_dir checkpoint.pth --learning_rate 0.001 --drop_p 0.2 --epochs 30 --arch alexnet --gpu y ```

In addition to get help execute:

`$ python train.py -h `

The **predict** app has the following arguments:
   
     1. Data image path                           --data_dir with default value 'flowers'
     2. checkpoint path                           --checkpoint
     3. Top K probabilities                       --top_k
     4. Categroy Names                            --category_names
     5. GPU to specified gpu resources use        --gpu with default value 'y'

Example to use the train classifier and predict a flower image, showing the top 3 possible flowers and no GPU usage:

```$ python predict.py flower checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu n```

In addition to get help execute:

`$ python predict.py -h `

Ouput example:

![Example](https://github.com/rafaelmata357/Flower-image-classifier/blob/master/output.png)

# Python version:
This app uses **Python 3.8.1**

# Libraries used:

- time 
- numpy 
- json
- PIL 
- os
- torch
- torch
- torchvision

# License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
