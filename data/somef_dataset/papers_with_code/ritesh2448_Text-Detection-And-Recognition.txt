# Text-Detection-And-Recognition

## Preprocessing
#### ICDAR dataset was used consisting of 461 images : https://drive.google.com/file/d/1ObrV9pbH_-LBGbIodWgB6W4dtQloTTH6/view
#### In-consistent File names were corrected 
#### Images and ground truth annotations were preprocessed and normalized as specified in : https://arxiv.org/pdf/1506.02640.pdf

## Text-Detection
#### Single class custom YOLO model was built for detecting text in images using tensorflow and keras
#### Predictions of the model were interpreted and bounding boxes were filtered using non max suppression

## Text-Recognition
#### Images were cropped using predicted bounding boxes and preprocessed using thresholding
#### The text is then recognized from the cropped images

## Web Application
#### Flask app is created to deploy the model
#### Website can be found at : https://tdar1234.herokuapp.com/

### Results:
![image](https://user-images.githubusercontent.com/63907547/115863536-372f0000-a453-11eb-9bfe-fc882c743e5f.png)
![image](https://user-images.githubusercontent.com/63907547/115863945-d18f4380-a453-11eb-8fe1-af37a838bf9b.png)






