# MaskRCNN + Faster RCNN 
##### (Personal Notes)
Mask RCNN implementation in PyTorch

Faster RCNN works in two phases:
1. Region proposals
2. Classifying regions

### Region Proposals
A region is an area of the original picture which might contain an object. Also known as *Region Of Interest (**RoI**)*

These are the most important aspects of an RCNN. They are also a source of bottlenecks. 

[**RCNN**](https://arxiv.org/abs/1311.2524) used a large number of region proposals by running it through a variety of category independent region proposal algorithms. These regions are then passed through a CNN. 

[**Fast RCNN**](https://arxiv.org/abs/1504.08083) is an improvment over **RCNN**. Instead of running the region proposal algorithms over the underlying image, the algorithms are run over a feature map. This feature map is obtained by passing the image through the Convlutional layers of any CNN. **Fast RCNN** is computationally less expensive when compared to RCNN. 

### Dataset
[Coco](http://cocodataset.org/#home)

### Paper
[RCNN](https://arxiv.org/abs/1311.2524)<br>
[Fast RCNN](https://arxiv.org/abs/1504.08083)<br>
[Mask RCNN](https://arxiv.org/abs/1703.06870)<br>