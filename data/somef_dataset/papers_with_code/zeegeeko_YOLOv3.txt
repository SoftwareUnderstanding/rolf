# Object Detection - YOLOv3

### YOLOv3 (You Only Look Once) implemented from scratch in TensorFlow.  

See the YOLOv3 Jupyter notebook in this repository for simple demos.  

![](./output/video_test.gif)

![](./output/detection_test2.jpg)

## Citing
arXiv paper, from the creators of YOLO: Joseph Redmon, Ali Farhadi.  
Redmon, Joseph, and Ali Farhadi. "YOLOv3: An Incremental Improvement." 2018. [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)

YOLOv3 implementation in TF-Slim  
Kapica, Pawel. "implementing YOLOv3 in TF-Slim." 2018 [https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)

YOLOv3 implementation in Pytorch  
Kathuria, Ayoosh. "How to implement a YOLO (v3) object detector from scratch in PyTorch." 2018 [https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

## Completed and TODOs
Completed:  
* YOLOv3 architecture in TensorFlow
* convert_weights() function to convert the Official YOLOv3 weights pre-trained on COCO dataset as a list of TensorFlow tf.assign operations. See the jupyter notebook on how to save it as a TensorFlow checkpoint.
* Image and Video post processing functions. (Drawing bounding boxes etc.)  
* Basic image and video functionality demo (jupyter notebook)

TODOs:
* Need to implement training and training pipeline to train on custom datasets.
* Improve performance of video processing.
* Improve bounding box and text drawing on images and video. (Looks pretty rough right now.)
* Add command line functionality for predictions.
