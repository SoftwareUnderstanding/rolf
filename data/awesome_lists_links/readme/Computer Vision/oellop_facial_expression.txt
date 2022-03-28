## INTRODUCTION 

This project aims to detect emotions on faces. First it detects faces with haarcascade classifier or with MTCNN model, then through a convolutionnal network it classifies the emotion from seven categories {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}. To train the model we had two choices of dataset : FER2013 and CK+48. In this version, the model is trained with FER2013 dataset and it uses MTCNN model to detect faces.


## REFERENCES 

Simonyan, Karen and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv, 4 Sept. 2014, arxiv.org/abs/1409.1556.

Goodfellow, Ian J., et al. "Challenges in Representation Learning: A report on three machine learning contests." arXiv, 1 July 2013, arxiv.org/abs/1307.0414.

Zhang, Kaipeng, et al. "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks." arXiv, 11 Apr. 2016, doi:10.1109/LSP.2016.2603342.

Viola, P. and M. Jones. Rapid object detection using a boosted cascade of simple features. IEEE, 2001, doi:10.1109/CVPR.2001.990517.

## SOURCES 

FER 2013 dataset :
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

CK+48 dataset :
https://www.kaggle.com/shawon10/ckplus

MTCNN :
https://github.com/ipazc/mtcnn

Haarcascade classifier :
https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html

