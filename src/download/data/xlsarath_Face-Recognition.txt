# FaceRec
A simple working facial recognition program.

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



@Author: David Vu

## Credits:
    -  Pretrained models from: https://github.com/davidsandberg/facenet
