# darknet-modify
## Introduction
This is a implemention of the customization yolov3 object detection algorithm based on darknet framework. Features are listed bollow.
* 1st: Adding the Attributes classification based on the Pre_box, such as overlapping, head or tail, day or night, etc. 
* 2nd: Several image augmentation schemes have been added, such as: image block overlap, to make yolo higher recall. Ground glass effect is applied to make the edge of object more stable and steady.
* 3rd: Based on the existing model, write the bounding boxes of the objects in a image into the xml file to assist further annotation.
we provide all related source code, and corresponding executable file can be generated.
For complete darknet code click here:https://pjreddie.com/darknet/install/
###### original paper:
* https://arxiv.org/abs/1804.02767
## 
## Usage
* To use Attribute function, please open macro "OPEN_OCC_CLASS_FLAG", in the project Scope.
* To use model select function in the Test stage, set field "-test_mode" to 0, 1 or 2, mean run image in a folder to get pre_box, select models base their recall and precision, respectively.
* To get objects' bounding box and save them to a xml file, please add field "-save_xml" in the script fild.
eg:
>@echo off\
>.\darknet_old.exe detector test E:/ E:\ E:\ -save_xml -test_mode 2 -dont_show\
>pause

