# Face-Mask-Detection
Face Mask Detection system is inspired from Deep Learning and Computer Vision concepts and aims at detecting face masks in static images as well as real time Video streams with help of libraries like OpenCV and Keras/Tensorflow.

The project comprises of two stages:

1.	Training the Face Mask Detector

2.	Applying the Face Mask Detector on static images and real-time video streams.

The dataset used for training the face mask detector is prepared by Prajna Bhandary and is available at Github. It comprides of 1,376 images belonging to two classes,namely with_mask(690 images ) and without_mask(686 images). 

The Face Mask Detection is divided into three Python scripts:

1.	mask_detector_training: It uses the MobileNetV2 architecture to fine tune the input dataset in order to create the desired model.

2.	img_mask_detection: It aims at detecting face masks in static images

3.	video_mask_detect: It detects face masks in real-time video stream.

The model accuracy achieved is approximately 97% on the test set.

Credits & Links:

Dataset: https://github.com/prajnasb/observations/tree/master/experiements/data

MobileNetV2 Architecture: https://arxiv.org/abs/1801.04381

PyImageSearch: https://www.pyimagesearch.com/

Motivated by the works of,

Adrian Rosebrock

Prajna Bhandary
