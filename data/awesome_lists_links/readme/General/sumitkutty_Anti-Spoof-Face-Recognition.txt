# Anti-Spoof-Face-Recognition
This project addresses a computer vision problem involving face recognition and anti-spoofing methods.

## Objective:
#### The goal is to build a face recognition system with an anti-spoof feature. The anti-spoof feature in this project is eye-blink detection. The condition for success is the detection of 5 blinks after a face is recognized.This way, a photograph cannot be used to get past the system.


## Dataset:
#### The dataset is created by collecting 10-15 pictures of one self and storing it in a folder under the name of the person whose picture it contains. All the folders are stored inside a unified dataset folder.


## Packages and Dependancies:
* Python 3.8
* OpenCV
* pickle
* imutils
* dlib
* face_recognition
* time
* numpy
* scipy
#### The rest of the dependancies are listed in the requirements.txt file. It can be installed from the command-line using 'pip'

## Training:
* The training is based on deep metric learning. This involves comparing the the embeddings of a face in the stream to the embeddings of all the faces saved during training. The closest estimated face is given as the output. 
* The training uses the famous ResNet-34 network from the 'Deep Residual Learning of Image Recognition' paper. Albeit, a pre-trained ResNet network with 29 layers and half the filters as the original one was used in the project.
* Basically, the pre-trained model is part of the face_recognition module and can be accessed from there. 
* The face was detected using a CNN that was part of the face_locations function.
* The labels and the face encodings during training are stored in a pickle object.


## Method:
* The method involves looping through the video and preprocessing the frames by converting to RGB, resizing the RGB image to the frame's dimensions.
* The faces are detected in the frame and stored in an array.
* The encodings for the detected faces in the stream is estimated and compared to the encodings from training, and the one with the maximum count is outputted.



## Anti-Spoof System:
* The eye-blink detection involves detecting the face and extracting the eyes and calculating the eye-aspect-ratio (EAR).
* The EAR basically represents the height-width ratio. As the eye blinks, the height value becomes small and the eye-aspect-ratio goes small.
* The EAR is calculated as the average of the EARs of both eyes.
* A threshold is set for the EAR, and if the EAR goes below the threshold, a blink is registered.
* If more than 5 blinks are registered, the system goes through. 

## Conclusion:
#### The system works with great accuracy and can be used in non-military grade sectors and employment centres in order to login into a system. The result of the system can be found in the output folder under this repo.


## References;
* http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
* https://arxiv.org/abs/1512.03385
* https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
* http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf



# THE END
