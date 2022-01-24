# cyberbully-classification
Spring 2018 Deep leaning project - classification and detection of cyberbully actions on images.
This project classifies cyberbully images from non-cyberbully images. The dataset is not available because of ownership rights.
The model is based on VGG16 and was implemented using Pytorch and Python 3.5, it achieved a classification accuracy of 68%. you can read about VGG16 here https://arxiv.org/abs/1409.1556

# To classify an image using our pretrained model
 - Download the pretrained weights from: https://drive.google.com/open?id=1bZ5Qs_79DCJ2ssNcnS4HricXIFjP937f
 - create a folder named "weights" in the root of this project and put the downloaded weight in that folder
 - Download Pytorch and skimage
 - It is recommended you install in a virtual environment. I use https://www.anaconda.com/
 - Enter python main.py -p 1 -i my_image.jpg on the command line, where my_image.jpg is the image you want to classify
 - wait for the classification result
 
# To train a model using your dataset
 - create a folder named "data" in the root of this project
 - the folder should contain two folders named "train" and "val"
 - each of the two folders should contain folders of different categories of images. eg, in the "train" folder 
 you could have "gossiping", "slapping" ... folders, where each of those folders contain images belonging to that category. Same goes for val
 - on line 24 in load_data.py change the variable data_dir to the name of your dataset folder, which in this case would be "data"
 - in experiments.py, in the function save_checkpoint, on line 112 replace the second argument in torch.save() with "weights/best_model.pt"
 - where weights is your weights folder in the root directory and best_model.pt is the file your model weight will be saved into. 
 - create an empty best_model.pt file and put it in that folder if you don't have that file in their. 
 - uncomment line 56 in main.py if you are using GPU 
 - enter python main.py on the command line
 - hyperparameters you might want to modify - learning rate, learning rate decay, weight_decay - you can add weight_decay=0.0005 as a keyword argument on line 59 in main.py
 - if you don't want to decay the learning rate on line 60 in main.py, then comment line 82 in experiments.py in the function train_model
 - if you choose to decay the learning rate then do not do the above but mind the amount of your step_size in line 60 in main.py

# Detection
The object detection and labeling part of this project is incomplete at the moment, we had labeled our dataset and attempted using YOLOv3. This can be continued on line 30 in experiments.py and by going through the YOLO documentation https://pjreddie.com/darknet/yolo/
OR you can do it from scratch using R-CNN, Faster R-CNN or Mask R-CNN etc
