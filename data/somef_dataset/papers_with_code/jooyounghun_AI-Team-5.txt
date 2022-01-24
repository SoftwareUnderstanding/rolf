# AI-Team-5: Combine Two Paper using Python
Project building architecture combining R-CNN network in Faster R-CNN paper and Facenet
___
## Introduction
we propose some application providing face detection of specific person and face blurring of the others.
There is two Paper for achieving job of project application.

## Requirements

- Previous Code Analytics from other git-repository about FRCNN,MRCNN
- Build unique RCNN Network for mobile computing environment
- python 3.x

## Datasets - Imagenet

ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. Currently we have an average of over five hundred images per node. We hope ImageNet will become a useful resource for researchers, educators, students and all of you who share our passion for pictures.

<img src="images/imagenet.png"/>


### Train Image Dataset
- download dataset from this [link](http://www.image-net.org) and put it in this project

### Test Image Dataset
The test image dataset are sampled from this [link](http://www.image-net.org) and put ti in this project

## TODO
* Reading Papers(R-CNN, Fast R-CNN, Faster R-CNN, FaceNet) and Studying
* Analyze existing code from github "https://github.com/jooyounghun/tensorpack/tree/master/examples/FasterRCNN"
* Build RCNN architecture

## DONE
* "FaceNet: A Unified Embedding for Face Recognition and Clustering" [Paper](https://arxiv.org/pdf/1503.03832) 
* "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" [Paper](https://arxiv.org/pdf/1506.01497)
* "Mask R-CNN" [Paper](https://arxiv.org/abs/1703.06870)


> Architecture of Network(Faster R-CNN & Mask R-CNN)

 **Faster-RCNN**

- Key Architecture "Region Proposal Network"
<table>
  <tr>
    <td>
     <img src="images/architecture_of_frcnn.png"/>
    </td>
  </tr>
</table>

- Detail of RPN 
<table>
  <tr>
    <td>
      <img src="images/sample_view_of_frcnn.png"/>
    </td>
  </tr>
</table>
  
 
 **Mask-RCNN**

- RoI Align: 2d interpolation for high accuracy of segmentation
<table>
  <tr>
    <td>
     <img src="images/architecture_of_mrcnn.png"/>
    </td>
  </tr>
</table>

- Result view from paper
<table>
  <tr>
    <td>
      <img src="images/sample_view_of_mrcnn.png"/>
    </td>
  </tr>
</table>


## Reference
- Florian Schroff, Dmitry Kalenichenko, James Philbin. Google Inc. FaceNet: A Unified Embedding for Face Recognition and Clustering. arXiv:1503.03832, 2015
- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object
Detection with Region Proposal Networks. arXiv:1506.01497, 2016
- Kaiming He Georgia Gkioxari Piotr Dollar Ross Girshick. Facebook AI Research (FAIR). Mask R-CNN. arXiv:1703.06870, 2018
