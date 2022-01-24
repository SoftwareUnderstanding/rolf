# PMR_WEB_APP1.0

# poor_man_rekognition
GSOC project 2019

Medium blog: https://medium.com/@b216029 (refer this for detailed information about each use-case)
independent github code: https://github.com/backSpace001/poor_man_rekognition

Building a free version of Amazon rekognition with maximum possible feature during a 3 months’ time span.

Use-cases:

   	1.Face and Eye Detection using OpenCV - - completed 👍
	2.Facial recognition of a video using deep metric learning - - completed 👍
	3.Celebrity Recognition - - completed 👍
 	4.Object Detection - - completed 👍
	5.Read text in images - - completed 
 	6.Facial Analysis - - completed 👍
 		Sad
 		Happy
 		Angry
 		Disgust
 		Fear
 		Surprise
 		Neutral
 	7.Scene Detection - - completed 👍

Libraries required: (requirement.txt is already available in the repository) 

	OpenCV: For using cascade files
	Numpy: For array operations
	Matplotlib.pyplot : For plotting
	Pickle : For serializing and de-serializing Python object structures
	Keras : For importing neural network models 
	Tensorflow : For CNN’s architecture and training
	Cython :  To generate CPython extension modules
	Pillow : To load images from files
	Lxml : To use the ElementTree API
	Flask==1.1
	gunicorn==19.3
	werkzeug==0.15
	opencv
	tesseract=3.02
	numpy==1.11
	scipy==0.18
	scikit-learn>=0.18

5 simple steps to download this repo, run in your local server and work on it accordingly.

Step 1.
        Download or clone this repo.

Step 2.
	Create a bin Folder inside the repo and download this weights from the link and paste it inside this bin folder
	https://drive.google.com/drive/folders/1hUY_n_H7jhdL9Z8yKKHZFB0wILGW_prH?usp=sharing
	(note-it is very big so couldn't upload it in the github)
	
Step 3.
        Get the requirments by typing in the command.
        pip install -r requirements.txt
        
Step 4.
        You are good to go.
        RUN $python app.py
     
Step 5.
        Open http://127.0.0.1:5000/ in your browser


USE-CASES ARE:

1.Face and eyes detection using OpenCV:

OpenCV comes with a trainer as well as a detector. Here I have used OpenCV for detection and later in the project, I will use it to create an XML file of faces for Face recognition. OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc. Those XML files are stored in the Library/etc/haarcascades. In this part, I have used face cascade and eye cascade to detect face and eyes in an image. OpenCV uses a machine learning algorithm and it contains pre-trained cascade XML files which can detect a face, eyes, etc. This basically breaks the image into pixels and form blocks, it does a very rough and quick test. If that passes, it does a slightly more detailed test, and so on. The algorithm may have 30 to 50 of these stages or cascades, and it will only detect a face if all stages pass.
This technique works on the Viola-Jones Algorithm, which is a part of deep learning. This statement was said on the context of:- deep learning is a class of machine learning algorithm that learns in supervised (e.g., classification) and/or unsupervised (e.g., pattern analysis) manners.
This part of face detection is also used in facial recognition section and there I will use this the file as an unrecognized file to be saved in the database and to be used as another face with no name registered.

Example rectangle features shown relative to the enclosing detection window. The sum of the pixels which lie within the white rectangles is subtracted from the sum of pixels in the grey rectangles. Two-rectangle features are shown in (A) and (B). Figure © shows a three-rectangle feature, and (D) a four-rectangle feature.


2.Facial Recognition

I have used Deep Learning face recognition embedding. Here I am using deep learning and this technique is called deep metric learning.In deep learning typically a network is trained to:
1. Accept a single input image
2. And output a classification/label for that image
However, deep metric learning is different. Instead, of trying to output a single label (or even the coordinates/bounding box of objects in an image), instead of outputting a real-valued feature vector. For the dlib facial recognition network, the output feature vector is 128-d (i.e., a list of 128 real-valued numbers) that is used to quantify the face. Training the network is done using triplets:
Facial Recognition via Deep metric learning involves “triplet training step”

I have first created a database for the training set and encoded (128-d) each face image into a numpy array and turn it into an XML file. Second I have imported that trained XML file into the main script to detect and recognize a face.


3.Celebrity Recognition

This part is same as the above one the only reason I made it a different sector is because this feature is listed in Amazon’s rekognition project and as this is a similar project I have to add this additional name tag and create a whole new dataset consisting of many known actors.
Here I have also used deep metric learning techniques.


4.Object detection

Object Detection is the process of finding real-world object instances like car, bike, TV, flowers, and humans in still images or Videos. It allows for the recognition, localization, and detection of multiple objects within an image which provides us with a much better understanding of an image as a whole. It is commonly used in applications such as image retrieval, security, surveillance, and advanced driver assistance systems (ADAS).
I have performed this using YOLOv2 on an image and a video file. You only look once (YOLO) is a state-of-the-art, real-time object detection system. On a Titan X, it processes images at 40–90 FPS and has an mAP on VOC 2007 of 78.6% and an mAP of 48.1% on COCO test-dev. One can find all the details about YOLOv2 here:
https://arxiv.org/pdf/1612.08242.pdf
https://www.youtube.com/watch?v=NM6lrxy0bxs


5.Read text in images

Extraction of text from an image is a subpart of image processing and is called OPTICAL CHARACTER RECOGNITION (OCR). I have used Tesseract which is an OCR engine developed by Google. It supports Unicode and has the ability to recognize more than 100 languages.


6.Facial expression recognition

https://medium.com/@b216029/report-3-494b2fdbb179  (refer this for this part)


7.Scene detection

Citation - www.algorithmia.com
I have coded to implement my part so as to perform the task, all the data will be provided by algorithmia and can be seen in the algorithmia website itself. My code will mere be a bridge. 

[note-this part is not included in the web app because of some complexity]
Scene detection is used for detecting transitions between shots in a video to split it into basic temporal segments. It helps video editors to automate the process of quickly splitting videos in bulk rather than editing it frame by frame by hand.
To run scene Detection Follow this steps:

	1.Create an account on Algorithmia (includes 5,000 free credits each month).
	2.Go to your profile page, click the Credentials tab, and find your API key.
	3.Find a test video. You can use a public URL (prefer Vimeo over youtube), or upload one to their hosted data storage.
	4.Install the Python Algorithmia client using the command “pip install algorithmia“.
	5.Copy the sample code below, replace YOUR_API_KEY with your own key, and run it to extract the scenes from your video!

https://medium.com/@b216029/report-3-final-8ef2de33a0d7

Cheers..!
