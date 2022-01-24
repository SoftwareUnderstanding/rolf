# FaceRec
A simple working facial recognition program.


## Installation:
    1. Install the dependencies

    2. Download the pretrained models here: https://drive.google.com/file/d/0Bx4sNrhhaBr3TDRMMUN3aGtHZzg/view?usp=sharing
    
        Then extract those files into models

    3. Run main.py

## Requirements:
    Python3 (3.5 ++ is recommended)

## Dependencies:

    opencv3

    numpy

    tensorflow ( 1.1.0-rc or  1.2.0 is recommended )


## Howto:
    `python3 main.py` to run the program
    `python3 main.py --mode "input"` to add new user. Start turning left, right, up, down after inputting the new name. Turn slowly to avoid blurred images



        
### Flags:
   `--mode "input"` to add new user into the data set
    


## General Information:
Project: IoT Image Recognition

We are adapting this facial recognition project to be used as an attendance system. 

This project was inspired by the poor attendance system currently implemented in JCU. How many times have you missed attendance because you forgot to tap in during class? for your average student it’s probably more than you can count.

Wouldn’t it be nice if you could just walk in to the classroom and your presence alone would be enough to clock your attendance. This might be possible using face recognition. By having cameras scanning the faces of people in the classroom the attendance would all be automated.

It could potentially also stop people from being able to “tap and leave” by monitoring the presence of students throughout the entire lecture.

### Info on the models I used:

Facial Recognition Architecture: Facenet Inception Resnet V1 

_Pretrained model is provided in Davidsandberg repo_

More information on the model: https://arxiv.org/abs/1602.07261

Face detection method: MTCNN

More info on MTCNN Face Detection: https://kpzhang93.github.io/MTCNN_face_detection_alignment/

Both of these models are run simultaneouslyx

### Framework and Libs:

Tensorflow: The infamous Google's Deep Learning Framework

OpenCV: Image processing (VideoCapture, resizing,..)










## Credits:
	-  Original project from: David Vu https://github.com/vudung45/FaceRec
    -  Pretrained models from: https://github.com/davidsandberg/facenet
