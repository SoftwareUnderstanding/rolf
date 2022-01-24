# A Faster-RCNN pytorch implementation

The goal was to implement a Faster RCNN model from
https://arxiv.org/abs/1506.01497 . In a simple, reader friendly way.
The model is not optimized.

Supports Resnet50 and Vgg16.

TODO:
- Generic interface for dataset (right now it's limited to one of my non-public datasets, under VOC format)
- Remove anchors out of bonds
- Improve negative proposals sampling for fastrcnn loss
- Optimize anchors and ROI operations
- Switch to a more pytorch syntax, with less numpy
