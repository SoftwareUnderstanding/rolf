# Face Recognition with tracking feature.
A facial recognition program with:
 - tracking feature to improve speed of implementation.
 - email feature if intruder is detected.


## Installation:
    1. Install the dependencies

    2. Download the pretrained models here: https://drive.google.com/file/d/0Bx4sNrhhaBr3TDRMMUN3aGtHZzg/view?usp=sharing
    
        Then extract those files into models
        
    3. In terminal do, source ./venv/bin/activate 
        
        then navigate to FaceRecognition folder location

    4. Run main.py

## Requirements:
    Python3 (3.5 ++ is recommended)

## Dependencies:

    opencv3

    numpy

    tensorflow ( 1.1.0-rc or  1.2.0 is recommended )


## Howto:
    `python3 main.py` to run the program
    `python3 main.py --mode "input"` to add new user. Start turning left, right, up, down after inputting the new name. Turn slowly to avoid blurred images

        
## Flags:
   `--mode "input"` to add new user into the data set
   
   `--type "http:\\IpAddress:port\video"` to set video stream input from url. Default: 0 (Laptop camera)

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
    -  Pretrained models from: https://github.com/davidsandberg/facenet
    -  Implementation inspired from : https://github.com/vudung45/FaceRec
