# Mask_RCNN
Implementation of Mask R-CNN in Tensorflow 2.0

# Understand Mask RCNN
Mask RNN is a variant of Faster RCNN model [3]. Similar to Faster RCNN, Mask RCNN is composed 3 parts:
* CNN Backbone - VGG or ResNet is commonly implemented as backbone for RCNN-based models in order to extract image features
* Region Proposal Networks - a simple network composedo of Convolution layers and Fully Connected layers to propose regions (bounding boxes) for objects
* ROIAlign - In Faster RCNN, ROI Pooling is used to apply proposed regions from RPN on feature maps from CNN Backone and to pixek max pixels for object classification. However, for masking objects, ROI Pooling leads to masking alignment due to ROI Pooling's nature. Hence, in Mask RCNN, ROI Align is in use instead

### Datasets
#### * COCO 2017
##### Download COCO 2017 and install pycocotools to parse COCO annotations
```
pip3 install Cpython
cd data
./get_coco_2017.sh
```

### References
* [1] Mask RCNN, https://arxiv.org/pdf/1703.06870.pdf
* [2] Rich feature hierarchies for accurate object detection and semantic segmentation
Tech report (v5), https://arxiv.org/pdf/1311.2524.pdf
* [3] Towards Real-Time Object-Detection with Region Proposal Networks, https://arxiv.org/pdf/1506.01497.pdf

### To-Do:
- [x] Build dataloader
- [x] Build ResNet that supports both Pyramid Feature Network and the 4th stage of ResNet
- [x] Build Region Proposal Network
- [x] Build ROIAlign layer
- [ ] Build Proposal Layer using Non-Max-Suppression
- [ ] Build Detection Layer
- [ ] Build loss functions
