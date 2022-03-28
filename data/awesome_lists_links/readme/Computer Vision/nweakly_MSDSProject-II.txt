# Object Detection in the Home Video Security Systems

## Project overview

![Ring Doorbell](https://github.com/nweakly/MSDSProject-II/blob/master/Data/scene00106.png "Screen shot of the Ring Doorbell recording")

The goal of this project is to conduct a feasibility study of applying deep learning techniques for detecting objects in the video recordings of the home security systems, such as the Ring doorbell, in order to be able to eventually build a customizable home security system. There are many potential scenarios where a custom-tuned home security system could be useful, for example, to notify homeowners that their mail and packages have been delivered, to combat so-called “porch pirates”, to detect undesirable activity nearby (e.g., people displaying guns) or to let parents know that their children safely returned home after taking their family dog for a walk. 

The Ring doorbell records video clips when detecting motion within a predetermined perimeter. However, this motion sensor can be triggered not only by humans walking up to the door, but by wild and domestic animals, passing vehicles, etc. So, the first step of this project is using an algorithm capable of processing video feed in real (or near real) time to identify and classify objects, and then training the model to identify additional context-dependent objects in the video recordings (video feed).
 
For a more detailed explanation of the project please watch the video presentation https://www.youtube.com/watch?v=B24XlEfF-u4 . It contains a description of the project, some practical suggestions and lessons learned while working on this project.

## Technical Requirements and Dependencies
- Anaconda package (64-bit version) on Windows 10
- Python 3.5 or higher
- TensorFlow (GPU version preferred)
- OpenCV
- Cython extensions - Python to C compiler and wrapper to be able to call DarkNet C code from Python
- Jupyter Notebook
- DarkNet framework - original implementation of the YOLO algorithm written in C and CUDA by Joseph Redmon https://github.com/pjreddie/darknet
- Darkflow - package translating Darknet to TensorFlow
- cfg (configuration) and weights files for the YOLO model downloaded from https://pjreddie.com/darknet/yolo/
- highly recommended - a separate conda virtual environment (to resolve version conflicts for the deep learning libraries) and use Anaconda for installations
- GPU GeForce RTX 2070 used during model training process, GeForce GTX1050 for all other file processing.

For detailed installation instructions please refere to a post by Abhijeet Kumar (https://appliedmachinelearning.blog/2018/05/27/running-yolo-v2-for-real-time-object-detection-on-videos-images-via-darkflow/ ) or https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html .

## Project steps
### Data collection, EDA and Preprocessing
For this project, I assembled custom training and testing datasets using the following tools and data sources:
- video recordings (for testing and extracting still images) from a personal Ring device collected using DataCollection.ipynb ;
- static images (for training, testing, and presentation) extracted from the video recordings using VLC media player (for instructions see  https://www.raymond.cc/blog/extract-video-frames-to-images-using-vlc-media-player/);
- additional training images were scraped using Google image search using Training_data_collection.py script;
- video files preprocessed using DataPreprocessing.ipynb to decrease the size, discard audio and cut out necessary parts of the video recordings;
- additional training pictures of a crowbar were taken by the author of the project; 
- Annotating_images.py and Drawing_Boxes.py scripts were used to manually draw bounding boxes around crowbars (to train a custom model) and create xml files with image annotations; 
- additional data augmentation techniques were randomly applied to the training data set (rotation, flipping, scaling, translation, color saturation changes, and cropping) using Photoshop batch processing.  

The original video recordings from the Ring device have frame size 1920x1080 pixels with 15 frames per second rate.   In order to accommodate existing hardware better, the videos were downsized to 640x360 pixels while retaining 15.0 frames/second rate. 
The resulting set of training crowbar images collected from all sources and augmentation techniques applied includes 554 total images.

### Fitting a pre-trained model
Since the detection speed is a very important factor in processing security videos, among all available CNN approaches  I chose to use a one-stage detector model, namely the __YOLO ("You Only look Once") model__ originally introduced in 2015 in the paper written by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.  The updated YOLOv2 algorithm was translated to Tensorflow by Trieu H. Trinh and is available as an open source __darkflow package__ (https://github.com/thtrieu/darkflow). 

A test example of YOLOv2 pretrained model applied to a static image can be found at https://github.com/nweakly/MSDSProject-II/blob/master/YOLO_Model_Test.ipynb which successfully with 68.4% confidence identified a cat in the picture. 

Using YOLOv2 for predictions is easier accomplished through the command line interface, for example using the following command:

```
python flow --model cfg/yolov2.cfg --load bin/yolov2.weights --demo videofile.avi  --gpu 1.0 --saveVideo
```
 (notes:  before running this command, navigate to the darkflow master directory and make sure to download the cfg file and the weights file, for example from https://pjreddie.com/darknet/yolo/ maintained by the authors of the YOLO algorithm.  For more options refer to the official darkflow repository instructions https://github.com/thtrieu/darkflow )
 
 ![Ring Doorbell-2]( https://github.com/nweakly/MSDSProject-II/blob/master/Data/scene00376.png "Screen shot of the Ring Doorbell recording")

Applied to the videos (please see mp4 files in Data/Processed folder),  the YOLOv2 and its smaller modification YOLOv2 tiny showed good detection results for large objects in both normal and low light conditions as long as there is an unobstructed view of an object.   I was also able to reach 25-26 frames per second while processing videos on GeForce GTX1050  and above 34 frames per second on GPU GeForce RTX 2070 for the full YOLOv2 (74 frames per seconds for YOLOv2-tiny), all of which are higher than 15 per second used in the Ring video recordings and is sufficient to process real-time surveyance video. 

### Training a New Model on a Custom Data Set
Next, I used the transfer learning approach, or a technique when a machine learning model trained to complete one task is repurposed to accomplished a different related task.  Due to the resource limitations, I chose a modification of the YOLOv2 model called YOLOv2-tiny as the pre-trained basis and changed its last two layers in order to train a new model on a custom dataset of crowbar pictures.
This required adjustments to the copy of .cfg file created for the new model (keep the original cfg file intact):
- in the last  [region] layer set the number of layers the model is  training for to 1:

```
[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
bias_match=1
classes=1
coords=4
num=5
softmax=1
```

- in the second to last [convolutional] layer, set the number of filters to 30 (num*(number of classes+5)=5*(1+5)=30:

```

[convolutional]
size=1
stride=1
pad=1
filters=30
activation=linear

[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52

```

- change the labels.txt file (saved in the darkflow master directory) to reflect the only class we are training for:

```
crowbar
```

Next, train the new model by using the following command and referencing the new cfg file, the weights file for the pretrained model and folders for the training images and corresponding annotations:

```
python flow  -- model cfg/yolov2-tiny-1c-4.cfg  --load bin/yolov2-tiny.weights --train --annotations new-model/annotations --dataset new_model/new_data --gpu 1.0
```
note: --gpu 1.0 parameter means that the training will be conducted 100% on GPU 

After training is complete, it is useful to save the graph and weights to protobuf file (.pb):

```
python flow  --model cfg/yolov2-tiny-1c-4.cfg  --load bin/yolov2-tiny.weights  --savepb
```
This command will generate .pb and .meta files that contain all the information necessary to make predictions using the newly trained model( using --pbLoad and --metaLoad instead of the --model and --load parameters in the demo example above).

### Use  the new model for predictions
Finally,  I used the newly trained model on previously unseen still images and videos.  Several images from the same folder can be forwarded for predictions at the same time using the following command:

```
python flow --imgdir new_model/test_pics  --pbLoad built_graph/yolov2-tiny-1c-4.pb --metaLoad built_graph/yolov2-tiny-1c-4.meta  --gpu 1.0 
```

Notes: 
- by default, the output images with bounding boxes are saved in a new subfolder out in the same folder ( new_model/test_pics/out );
- add --json if you would like to generate output json files with the pixel location for each bounding box.

I also tested the model on the videos containing crowbars with some neutral backgrounds and video files from the Ring doorbell.  
```
python flow  --pbLoad built_graph/yolov2-tiny-1c-4.pb --metaLoad built_graph/yolov2-tiny-1c-4.meta  --demo new_model/test_video/IMG_0851.MOV --threshold 0.67 --gpu 1.0 --saveVideo
```

Please see examples of the results in  https://github.com/nweakly/MSDSProject-II/tree/master/new_model/test_video and  https://github.com/nweakly/MSDSProject-II/tree/master/new_model/test_ring_video . 

Note: darkflow also allows using information about different checkpoints generated during training to produce predictions. In many cases it is useful to compare results at the different stages, however, checkpoint files are not included in this repository due to space restrictions.

## Conclusions
- YOLOv2  and YOLOv2-tiny are very fast models capable of processing real-time video stream and providing reliable detection results for large objects with an unobstructed view;
- Detection accuracy decreases for smaller objects and partially visible (obstructed view objects);
- It is possible to create custom -built object detection systems for video surveillance using the YOLOv2 model and its tiny modification as demonstrated by the Ring example;
- The loss function is an important training indicator, but a lower loss does not necessarily guarantee a better model.  When training on a dataset of images scraped from Google image search, I reached the lowest moving average loss of all my model - 0.5441. However, the model had a very low recall. Training on a dataset of still images extracted from the Ring videos resulted in a moving average loss of 0.6824, however, the model had a high false positive rate. It was incorrectly identifying crowbars where they were previously present in training images, clearly picking up on some other features.  When training on a more balanced set, the moving average loss was fluctuating between 0.9 and 1.01 and not decreasing any further.  This model turned out to provide better detection results. 
 - However, increasing accuracy would require a more thorough training process and better training dataset (training exclusively on images collected from the internet resulted in prediction confidence of only 2-5%; using many still images extracted from the Ring videos resulted in an overfitted model which was "detecting" non-existing crowbars in the locations previously seen on training photos and not detecting objects in the previously unseen locations; combining training sets yielded the best results);
- While labeling the training images, I did not specify if a bounding box included a whole crowbar or its partial image, as a result, parts of the same crowbar in test videos were occasionally detected as two separate objects;
- YOLO algorithm can be used in video security systems to trigger an alarm in case of a particular event but detecting small objects and tracking the same object from frame to the frame would require using different approaches;
 - Training a custom deep learning model is not only science but it can also feel as an art requiring a lot of patience and experience as well. While working on this project, I experimented with training six different models (different training sets, size of training images,  learning rates, number of epochs, batch sizes) with the longest training process running on GeForce RTX 2070 GPU for 46 hours and the results could not easily be predicted in advance.

## References and Links
- Video presentation for this project: https://www.youtube.com/watch?v=B24XlEfF-u4

- Darkflow library (Darknet translated to TensorFlow) https://github.com/thtrieu/darkflow
- Darknet framework https://github.com/pjreddie/darknet
- Darknet project site:  https://pjreddie.com/darknet/yolo/ . Use to download configuration and pretrained weights files.
- Jay, M. Series of YOLO tutorials: https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM&index=1 and 
https://github.com/markjay4k/YOLO-series 
- Instructions for setting up YOLO using Anaconda and Windows https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html
- Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2015). You Only Look Once: unified, real-time object detection. Retrieved from:  https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
- Redmon, J., Farhadi, A. (2016). YOLO9 000: Better, Faster, Stronger. Retrieved from https://arxiv.org/pdf/1612.08242v1.pdf


