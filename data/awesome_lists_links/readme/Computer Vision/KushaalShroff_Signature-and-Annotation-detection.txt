# Faster R-CNN for Signature and Annotation detection using Keras

The original code of Keras version of Faster R-CNN I used was written by yhenon (resource link: GitHub .) He used the PASCAL VOC 2007, 2012, and MS COCO datasets. I applied configs different from his work to fit my dataset and I removed unuseful code.

## Project Structure
Use the sign_detection_train_vgg.ipynb file to train on any dataset of your choice. Define the annotaion and bounding box coordinates in the annotaion.txt file. It uses a VGG 16 model. For future scope You can add RESNET and other models. Use the sign_detection_test_vgg.ipynb file to test your images. During Training we keep updating and saving the weights, incase of any system failure or power cut, our trained data would still be saved to the nearest epoch.

## Requirements
1) python 3.6+ Link to download and install (https://www.python.org/downloads/)
2) You will need jupyter notebook to open the .ipynb files. (On command line type pip install jupyter)
3) Tensorflow(if you have gpu the you could use a tensorflow-gpu version). (On command line type pip install tensorflow or tensorflow-gpu)
4) Keras (On command line type pip install keras)
5) Numpy (On command line type pip install numpy)
6) Pandas (On command line type pip install pandas)

## Introduction to Faster-RCNN 
Faster R-CNN has two networks: region proposal network (RPN) for generating region proposals and a network using these proposals to detect objects. The main difference here with Fast R-CNN is that the later uses selective search to generate region proposals. The time cost of generating region proposals is much smaller in RPN than selective search, when RPN shares the most computation with the object detection network. Briefly, RPN ranks region boxes (called anchors) and proposes the ones most likely containing objects.

## Regional Purpose Network
The output of a region proposal network (RPN) is a bunch of boxes/proposals that will be examined by a classifier and regressor to eventually check the occurrence of objects. To be more precise, RPN predicts the possibility of an anchor being background or foreground, and refine the anchor.

<img src="fasterrcnn.jpeg"
     style="float: left; margin-right: 10px;" />

## Classifier
The first step of training a classifier is make a training dataset. The training data is the anchors we get from the above process and the ground-truth boxes. The problem we need to solve here is how we use the ground-truth boxes to label the anchors. The basic idea here is that we want to label the anchors having the higher overlaps with ground-truth boxes as foreground, the ones with lower overlaps as background. Apparently, it needs some tweaks and compromise to seperate foreground and background. You can check the details here in the implementation. Now we have labels for the anchors.

## References 
1. Fast R-CNN: https://arxiv.org/pdf/1504.08083.pdf
2. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks: https://arxiv.org/pdf/1506.01497.pdf
3. py-faster-rcnn: https://github.com/rbgirshick/py-faster-rcnn
4. A guide to receptive field arithmetic for Convolutional Neural Networks: https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
5. Region of interest pooling explained: https://blog.deepsense.ai/region-of-interest-pooling-explained/
