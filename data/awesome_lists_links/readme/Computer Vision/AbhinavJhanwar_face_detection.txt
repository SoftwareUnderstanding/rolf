I have implemented following face detection models here namely-
1) yolov3
2) resnet
3) dlib hog
4) dlib cnn
5) opencv

# 1. YOLO V3
* **Paper**- https://arxiv.org/pdf/1804.02767.pdf
* **Repo**- https://github.com/sthanhng/yoloface
* **References**- https://towardsdatascience.com/review-yolov3-you-only-look-once-object-detection-eab75d7a1ba6
* **Methodology/framework**- It basically uses darket-53 pretrained on imagenet dataset as feature extractor and then another 53 layers for detection/classification. Detections are performed on 3 different scales like SSD to capture all the objects of various sizes. Here object classification is basically done through logistic regression and not softmax so that objects like person/boy are both detected
* **Output**- Shape of the detection kernel is 1 x 1 x (B x (5 + C) ). Here B is the number of bounding boxes a cell on the feature map can predict, "5" is for the 4 bounding box attributes and one object confidence, and C is the number of classes
* **Architecture/base network**- Darknet-53 as feature extractor trained on imagenet data, another 53 layers for classification/detection of objects
* **Dataset trained on**- WIDER FACE: A Face Detection Benchmark
* **Input image size**- 416x416
* **Anchor boxes**- 9 anchor boxes generated using K-Means clustering (3 for each scale)
## Usage-
1) Download weights from the link in the following repository and save in folder yolo - https://github.com/sthanhng/yoloface
2) modify config.conf file as following-
* **model**- yolov3
* **source_type**- webcam or video or image
* **source_path**- camera id if using 'webcam' or video/image path otherwise
* **output_dir**- directory to save detected video/image
3) run python face_detection.py

# 2. RESNET
* **Repo**- https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
* **Methodology/framework**- This basically works on SSD framework with ResNet architecture
* **Input image size**- 300x300
## Usage-
1) Download weights from the following repository and save in folder resnet - https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
2) modify config.conf file as following-
* **model**- resnet
* **source_type**- webcam or video or image
* **source_path**- camera id if using 'webcam' or video/image path otherwise
* **output_dir**- directory to save detected video/image
3) run python face_detection.py

# 3. dlib cnn/hog-
* **Repo**- https://github.com/ageitgey/face_recognition
## Usage-
1) Install the required libraries-<br>

**for ubuntu-**
```
pip install face_recogntion
pip install -r requirements.txt
```
**for windows-**
```
pip install dlib>19.7
pip install face_recognition
pip install -r requirements.txt
```
2) modify config.conf file as following-
* **model**- dlib_hog or dlib_cnn
* **source_type**- webcam or video or image
* **source_path**- camera id if using 'webcam' or video/image path otherwise
* **output_dir**- directory to save detected video/image
3) run python face_detection.py


# 4. opencv-
## Usage-
1) install the required libraries-
```
pip install -r requirements.txt
```
2) modify config.conf file as following-
* **model**- opencv
* **source_type**- webcam or video or image
* **source_path**- camera id if using 'webcam' or video/image path otherwise
* **output_dir**- directory to save detected video/image
3) run python face_detection.py

**Some interesting findings-**<br>
check the prediction on Abhinav2.png with resnet and yolo. it will definitely surprise you.
