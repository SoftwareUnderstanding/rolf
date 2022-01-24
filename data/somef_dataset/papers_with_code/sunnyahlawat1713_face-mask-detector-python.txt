# face-mask-detector-python
In this project, I have built a program in python to detect people's faces in a static image/video stream and predict whether the person is wearing a mask or not.

# Face detection:
For face detection, I have used OpenCV's readNet(cv2.dnn.readNet()) model which takes in two arguments, a binary file containing trained weights, and a text file containing network configuration. Both of these files can be found in the face_detector folder of this repository. 

# Mask detection: 
For mask prediction(on detected faces), I have used the MobileNetV2 model trained on ImageNet weights with the top of the model excluded and the weights of the remaining layers have been retained. The top(head) of the model has been re-defined to suit the specific classification problem at hand(mask-detection).

# Dataset:
The dataset has been taken from Adrian Rosebrock's tutorial linked below:

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

The dataset can be accessed by navigating to the Downloads section of the page. This dataset mainly contains images with white masks, so I've also taken some images with black masks from the internet. 

# Using the scripts:

Create a folder for this project. Copy the three python script files into that folder along with the face_detector folder provided in the repository. Create a folder called "dataset" with two sub-folders called "with_mask" and "without_mask" . Copy the corresponding images from the dataset into these folders. 

# 1. Training the model:

Open up a terminal (preferably a conda env prompt with all the dependencies installed) and then change the directory to wherever the master folder is situated. To train the model, type:

python train_mask_detector.py --dataset dataset

The model should train with a plot for training loss and accuracy generated at the end along with a classification report with information such as precision and recall for both classes. My computer does not have a GPU so I've only trained it for 10 epochs but those with GPUs can train further (e.g. 20 epochs) . I achieved an accuracy of more than 95% on the validation set but that can be definitely increased by training for longer amounts of time. 

# 2. Classifying static images:

Now that our model has been trained, we can classify images to check whether the person in the image is wearing a mask. To do so, type the following command in the terminal: 

python detect_mask_image.py --image path_to_image

# 3. Detecting and classifying in a video stream:

Currently, the script leads to taking input from webcam for video stream. The command line is:

python detect_mask_video.py 

To quit the stream, press "Q" on your keyboard. 

Notes: Currently, the dataset on which the provided mask_detector.model file has been trained on, has a disproportionately higher number of white masks than black ones. I will correct this problem in the near future. 

Link for MobileNetV2 research paper:

https://arxiv.org/abs/1801.04381

I learnt a lot from Adrian Rosebrock's tutorial on this topic, the link to which has been provided in the dataset section
