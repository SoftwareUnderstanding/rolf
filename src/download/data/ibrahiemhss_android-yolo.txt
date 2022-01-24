# TensorFlow YOLO object detection on Android



**android-yolo** is the first implementation of YOLO for TensorFlow on an Android device. It is compatible with Android Studio and usable out of the box. It can detect the 20 classes of objects in the Pascal VOC dataset: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train and tv/monitor. The network only outputs one predicted bounding box at a time for now. The code can and will be extended in the future to output several predictions.

To use this demo first clone the repository. Download the TensorFlow YOLO [model](https://drive.google.com/file/d/0B2fFW2t9-qW3MVJlQ29LRzlLT2c/view?usp=sharing) and put it in android-yolo/app/src/main/assets. Then open the project on Android Studio. Once the project is open you can run the project on your Android device using the Run 'app' command and selecting your device.
![Screenshot](Selection_035.png)
![Screenshot](Selection_036.png)
![Screenshot](Selection_037.png)

