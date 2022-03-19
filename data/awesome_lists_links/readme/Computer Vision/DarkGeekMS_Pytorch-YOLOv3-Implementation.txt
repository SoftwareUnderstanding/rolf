# YOLOv3 Pytorch Implementation

### Brief Description 

This is an implementation of the object detection architecture "YOLOv3" using the deep learning framework "Pytorch".
YOLOv3 is a neural network architecture for performing object detection on images and videos, it's a refinement of the previous 
YOLO9000.


### Utilized Technologies and Frameworks

- Pytorch.
- Basic Python Data Manipulation Packages (Matplotlib,....etc).

### Repository Structure

**1) Main Files:**
- **yolov3_model.py:** contains the definition of YOLOv3 architecture.
- **yolov3_train.py:** contains the training setup of the network.
- **yolov3_validate.py:** contains the validation setup to test your trained model.
- **yolov3_detect.py:** contains the setup to run inference on images using the trained model.

**2) Configs:**
- **yolov3.cfg:** contains the configuration of the neural network architecture along with hyperparameters.
- **yolov3.data:** contains the paths to the data and number of classes.
- **yolov3.names:** contains the names of the classes.

**3) Data Preparation:**
- **img_prep.py:** used to prepare the dataset images for the model.
- **label_prep.py:** used to prepare the labels (bounding boxes) for the model.

**4) Utilities:**
- **data_classes.py:** contains the classes for the dataloader.
- **helper_func.py:** contains some functions to parse config files.
- **yolov3_utils.py:** contains some utility functions: non-max suppression, output preparation and some metrics.

### How to Use it 

- Run data preparation files providing train path, labels path and test path.
- Edit config files at the mentioned lines beginning with (//--------->).
- Optionally, Edit some hyperparameters in yolov3.cfg.
- Run yolov3_train.py for training.
- Run yolov3_validate.py for validation.
- Run yolov3_detect.py for Inference.

### Acknowledgement

- "YOLOv3: An Incremental Improvement" Research Paper: https://arxiv.org/abs/1804.02767
- I have learned a lot and utilized a lot of resources from this repository:
https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch


Thanks a lot ^_^
