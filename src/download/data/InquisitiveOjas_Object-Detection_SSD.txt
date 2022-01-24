# Object-Detection_SSD
Learning attempt on object detection using Single Shot Multi Box Detection.

SSD is the most powerful algorithm as compared to faster-R-CNN or YOLO.

This repository includes:
                          
                          (I)Detection of Dogs jumping on a ground.
                          (II)Detection of Horses galloping in one of the most magical places in the world. 

SSD will divide the image into segments. Make boxes around these segments. Each box individually predicts whether the object is present in it or not.
Also predicts where the object is present in the image.

Faster R-CNN uses a region proposal network to create boundary boxes and utilizes those boxes to classify objects. 
While it is considered the state-of-the-art in accuracy, the whole process runs at 7 frames per second.Far below what a real-time processing needs. 
 
SSD speeds up the process by eliminating the need of the region proposal network.
 
To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes. 
These improvements allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher.

The SSD object detection composes of 2 parts:

                                            (1)Extract feature maps
                                            (2)Apply convolution filters to detect objects.

SSD uses VGG16 to extract feature maps. Then it detects objects using the Conv4_3 layer. 

For illustration, we draw the Conv4_3 to be 8 × 8 spatially (it should be 38 × 38). For each cell (also called location), it makes 4 object predictions.Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object), and we pick the highest score as the class for the bounded object. Conv4_3 makes a total of 38 × 38 × 4 predictions: four predictions per cell regardless of the depth of the feature maps. As expected, many predictions contain no object. SSD reserves a class “0” to indicate it has no objects.

SSD does not use a delegated region proposal network. Instead, it resolves to a very simple method. It computes both the location and class scores using small convolution filters. After extracting the feature maps, SSD applies 3 × 3 convolution filters for each cell to make predictions. (These filters compute the results just like the regular CNN filters.) Each filter outputs 25 channels: 21 scores for each class plus one boundary box.

It uses multiple layers (multi-scale feature maps) to detect objects independently. As CNN reduces the spatial dimension gradually, the resolution of the feature maps also decrease. SSD uses lower resolution layers to detect larger scale objects. For example, the 4× 4 feature maps are used for larger scale object.

SSD adds 6 more auxiliary convolution layers after the VGG16. Five of them will be added for object detection. In three of those layers, we make 6 predictions instead of 4. In total, SSD makes 8732 predictions using 6 layers.

Multi-scale feature maps improve accuracy significantly. 

SSD predictions are classified as positive matches or negative matches. SSD only uses positive matches in calculating the localization cost (the mismatch of the boundary box). If the corresponding default boundary box (not the predicted boundary box) has an IoU greater than 0.5 with the ground truth, the match is positive. Otherwise, it is negative. (IoU, the Intersection over Union, is the ratio between the intersected area over the joined area for two regions.)

Reference : https://arxiv.org/pdf/1512.02325.pdf
