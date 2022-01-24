## Honey-Bee-Images-Classfication
### Author: Xiaochi Ge, Phoebe Wu, Ruyue Zhang, Yijia Chen

The dataset is from kaggle, The BeeImage Dataset: Annotated Honey Bee Images: https://www.kaggle.com/jenny18/honey-bee-annotated-images.

It contains 5,172 bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.

#### Getting Started:
##### 1. Download the Data
Please download the dataset here:
http://storage.googleapis.com/group4_data/input.zip

Please check on the path before running the code. The dataset should be in the right path as '../input' after unzip the file.

To view an image, run show_img.py
 
##### 2. EDA

Please run EDA.py

##### 3. Modelling
We use CNN to classify bee subspecies and hive health by 2 frameworks Keras and Pytorch. 
   - Keras:  
     - Subspecies_Keras.py (20 cnn models included with different hyperparameters)
       - Focal loss paper  https://arxiv.org/abs/1708.02002
     - HiveHealth_Keras.py (comment out line 135, 136 to run training1, at the same time, comment line 138, 139)

   - Pytorch: 
     - Subspecies_Torch.py (you could try different channel in cnn module, and change batch size for experiment)
     - HiveHealth_Torch.py
    
