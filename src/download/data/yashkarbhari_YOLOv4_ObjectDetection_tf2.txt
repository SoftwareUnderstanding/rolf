# YOLOv4_ObjectDetection_tf2

This repository is cloned from https://github.com/theAIGuysCode/yolov4-deepsort

YOLOv4 is state of the art algorithm to detect objects.

DeepSORT - The most popular and one of the most widely used, elegant object tracking framework is Deep SORT, an extension to SORT (Simple Real time Tracker).

Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of object_tracker.py Within the list __allowed_classes__ just add whichever classes you want the tracker to track. The classes can be any of the 80 that the model is trained on, see which classes you can track in the file data/classes/coco.names.

Reference: YOLOv4 paper - https://arxiv.org/abs/2004.10934
