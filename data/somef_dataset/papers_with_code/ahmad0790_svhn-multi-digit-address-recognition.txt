# SVHN Multi Digit Address Recognition and Localization Using Convolutional Neural Networks

This project trains different Convolutional Neural Network model architectures using street house digit images in the Stanford SVHN dataset to detect the correct house address number in real life images and the location of the house address in an image (both the correct digits and also the correct sequencing of digits: For example: it can successfully differentiate between 1234 Main Street vs 1243 Main Street). 

The model is trained on 150K+ unique images total with data augmented through random rotations and shifts to help results be more robust to location, lighting, orientation and scale. After training a sliding window (but originally a gaussian pyramid approach) is used to detect the exact location of a number in an image in addition to the number itself

CNN Model architectures were based on the folloiwng academic papers:

1) `Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks
Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet`
(https://arxiv.org/pdf/1312.6082.pdf)

2) `VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION 
Karen Simonyan & Andrew Zisserman + Visual Geometry Group, Department of Engineering Science, University of Oxford` (https://arxiv.org/pdf/1409.1556.pdf)

Th Goodfellow paper performs slightly better coming at around 92% all digit accuracy (with the worst digit number no less than 95% accuracy). To showcase performance of classifier under different viewing conditions a short video has been uploaded on youtube for reference:<br/>

https://youtu.be/ln-Tg_ou5iU

Results are also summarized in report.pdf for reference.

## Important Files
### Data Processing
`load_data.py` Processes each image, crops image with bounding box and generates training labels for each image
`load_data_rotations.py` Same preprrocessing but with added random rotations<br/>
`load_data_negatives.py` Same preprrocessing but with negative labelled images added (Those imges with no street number)

### Model Architecture
`CNN.py` Builds the Goodfellow paper CNN model<br/>
`VGG_model.py` Builds the VGG paper CNN model

### Model Training
`model_augmented.py` Trains the Goodfellow based model on training data using the CNN.py model architecture<br/>

### Video Predictions and Localization
`predictions.py` Functions to make prredictions on new images<br/>
`create_video.py` Predicts for each frame in a video both the location and sequence of streeet numbers in an image. saves video as an mp4

## Analysis Report
`report_ahmad_khan.pdf` Report summarizes results and model performance using the different CNN model architectures

## Example Images
Here are some examples of street house images correctly classified by the model showing scale, rotation, positional and lighting invariance<br/>

A full video can also be found on youtube here: https://youtu.be/ln-Tg_ou5iU

![Image 1](graded_images/1.png)
![Image 2](graded_images/3.png)
![Image 2](graded_images/5.png)
