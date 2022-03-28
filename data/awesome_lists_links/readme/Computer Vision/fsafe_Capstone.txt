This project implementation is based on the following paper:

Youbao Tang, Ke Yan*, Yuxing Tang*, Jiamin Liu*, Jing Xiao, Ronald M. Summers, "ULDor: A Universal Lesion Detector for CT Scans with Pseudo Masks and Hard Negative Example Mining," ISBI, 2019<sup>**</sup> [(arXiv)](https://arxiv.org/abs/1901.06359) 
Some of the major difference between this project and the above paper are:
1. The paper employs a "Hard Negative Example Mining" method which is not currently implemented in this project
2. For the MaskR-CNN backbone this implementation uses a Feature Pyramid Network(FPN) with a ResNet-50 for the bottom-up pathway whereas the paper employs a ResNet-101 for the backbone of the MaskR-CNN.
3. A learning rate decay is not used in this implementation. 

In addition the data preprocessing steps were adapted from:
https://github.com/rsummers11/CADLab/tree/master/lesion_detector_3DCE

The above link is a code implementation of the following paper:
Ke Yan et al., "3D Context Enhanced Region-based Convolutional Neural Network for End-to-End Lesion Detection," MICCAI, 2018 [(arXiv)](https://arxiv.org/abs/1806.09648)  

# UNIVERSAL LESION DETECTOR FOR CT SCANS WITH PSEUDO MASKS

## Introduction:
Image recognition and deep learning technologies using Convolutional Neural Networks (CNN) have demonstrated remarkable progress in the medical image analysis field. Traditionally radiologists with extensive clinical expertise visually asses medical images to detect and classify diseases. The task of lesion detection is particularly challenging because non-lesions and true lesions
can appear similar. 

For my capstone project I use a Mask R-CNN <sup>[1](https://arxiv.org/abs/1703.06870)</sup> with a ResNet-50 Feature Pyramid Network backbone to detect lesions in a CT scan. The model outputs a bounding box, instance segmentation mask and confidence score for each detected lesion. Mask R-CNN was built by the Facebook AI research team (FAIR) in April 2017.

The algorithms are implemented using PyToch and run on an Nvidia Quadro P4000 GPU. 
## Data:
The dataset used to train the model has a variety of lesion types such as lung nodules, liver tumors and enlarged lymph nodes. This large-scale dataset of CT images, named DeepLesion, is publicly available and has over 32,000 annotated lesions.<sup>[2](https://nihcc.app.box.com/v/DeepLesion/)</sup> The data consists of 32,120 axial computed tomography (CT) slices with 1 to 3 lesions in each image. The annotations and meta-data are stored in three excel files:

DL_info_train.csv (training set)

DL_info_val.csv (validation set)

DL_info_test.csv (test set)

Each row contains information for one lesion. For a list of meanings for each column in the annotation excel files go to: 
https://nihcc.app.box.com/v/DeepLesion/file/306056134060 

Here is a description of some of the key fields:

column 6: Image coordinates (in pixel) of the two RECIST diameters of the lesion. There are 8 coordinates for each annotated lesion and the first 4 coordinates are for the long axis. "Each RECIST-diameter bookmark consists of two lines: one measuring the longest diameter of the lesion and the second measuring its longest perpendicular diameter in the plane of measurement."<sup>[3](https://nihcc.app.box.com/v/DeepLesion/file/306049009356)</sup> These coordinates are used to construct a pseudo-mask for each lesion. More details on this later. 

column 7: Bounding box coordinates which consists of the upper left and lower right coordinates of the bounding box for each annotated lesion.

column 13: Distance between image pixels in mm for x,y,z axes. The third value represents the vertical distance between image slices 

An important point to note is that the total size of the images in the dataset is 225GB however out of the 225GB there is only annotation (i.e. labelled) information for images totaling 7.2GB in size. In this implementation training was only done on a portion of the labelled data.
## Image Pre-Processing and Data Pipeline:
Several pre-processing steps were conducted on the image and labels prior to serving them to the model. These steps were placed in a data pipeline so that the same pipeline could be used during the training, validation and testing phase.

1. Offset Adjustment: Subtract 32768 from the pixel intensity of the stored unsigned 16 bit images to obtain the original Hounsfield unit (HU) values. The Hounsfield scale is a quantitative scale for describing radiodensity.
2. Intensity Windowing<sup>[4](https://radiopaedia.org/articles/windowing-ct)</sup>: Rescale intensity from range in window to floating-point numbers in [0,255]. Different structures (lung, soft tissue, bone etc.), have
different windows however a for this project a single range (-1024,3071 HU) is used that
covers the intensity ranges of the lung, soft tissue, and bone.
3. Resizing: Resize every image slice so that each pixel corresponds to 0.8mm.
4. Tensor Conversion: Covert image and labels to tensors
5. Clip black border (commented out in code): Clip black borders in image for computational efficiency and adjust bounding box and segmentatoin mask accordingly. For some unknown reason this transformation is apparently preventing the model's training loss to converge. This merits further investigation.

Psudo-Mask Construction:

The DeepLesion dataset includes a file containing bookmarks (i.e. bounding boxes and RECIST diameters) for each lesion which are marked by radiologists. However the dataset does not include a segmentation mask for each lesion. Therefore using the method explained in \(\**\) a psudo-mask is constructed by fitting an ellipse around the RECIST diameters. 


PyTorch has tools to streamline the data preparation process used in many machine learning problems. Below I briefly go through the concepts which are used to make data loading easy.

torch.utils.data.Dataset class:

This is an abstract class which represents the dataset. In this project the class DeepLesion is a subclass of the Dataset class. DeepLesion overrides the following methods of the Dataset class:

* \_\_len__ so that len(dataset) returns the size of the dataset.
* \_\_getitem__ to support the indexing such that dataset[i] can be used to get ith sample

A sample of the DeepLesion dataset will be a tuple consisting of the CT scan image (torch.tensor) and a dictionary of labels and meta data. The dictionary has the following structure:

* boxes : List of bounding boxes of each lesion in image
* masks : Instance segmentation mask for each lesion in image
* labels : List of 1's because 1 represents the label of the lesion class
* image_id : String storing the relative filename of image slice (e.g. 004408_01_02\\088.png)

DeepLesion's initializer also takes an optional argument 'transform' which is used to apply the preprocessing steps described above

Transformations: 

For each preprocessing/transformation step a separate class is created. These classes will implement a \_\_call__ method and an \_\_init__ method. The \_\_init__ is used to customize the transformation. For example in the ToOriginalHU class, 'offset' is passed to the \_\_init__ method. The \_\_call__ method on the other hand receives the parameters which are potentially transformed. In the  ToOriginalHU class the 'offset' value is subtracted from the image, which is passed as a parameter to the \_\_call__ method. This is what the resulting code looks like:

    class ToOriginalHU(object):
        """Subtracting offset from the16-bit pixel intensities to
        obtain the original Hounsfield Unit (HU) values"""
    
        def __init__(self, offset):
            self.offset = offset
    
        def __call__(self, image, spacing=None, targets=None, ):
            image = image.astype(np.float32, copy=False) - self.offset
            return image, spacing, targets
All such classes are placed together in a list and the resulting list is passed to the Compose class initializer

Compose class:

This class also has a an \_\_init__ method and a \_\_call__ method. The \_\_init__ method initializes the Compose class with a collection of other transformation classes initializers each representing a transformation as described above. The \_\_call__ method simply traverses the collection instantiating each transformation and storing the result of each transformation in the same variables which are passed as parameters to the next transformation. By doing this Compose chains the preprocessing steps together.

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
    
        def __call__(self, image, spacing=None, targets=None):
            for t in self.transforms:
                image, spacing, targets = t(image, spacing, targets)
            return image, spacing, targets
Now let's look at how these concepts are used in the project:

    from data import transforms as T
    data_transforms = {
        'train': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'val': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
        , 'test': T.Compose([T.ToOriginalHU(INTENSITY_OFFSET)
                            , T.IntensityWindowing(WINDOWING)
                            , T.SpacingResize(NORM_SPACING, MAX_SIZE)
                            , T.ToTensor()])
    }
    image_datasets = {x: DeepLesion(DIR_IN + os.sep + x, GT_FN_DICT[x], data_transforms[x]) for x in ['train', 'val'
                                                                                                      , 'test']}
The above code snippet first defines a data_transformation dictionary which has 'train', 'val' and 'test' as the key values and an instance of the Compose class (with all preprocessing steps) as the value for each key. Similarly the image_dataset is a dictionary with the same keys and the values contain an instance of the DeepLesion class. Note that an instance of the Compose class is passed to the 'transform' parameter (third parameter) to create an instance of the DeepLesion class. The value of the 'transform' parameter is stored in the 'self.transform' attribute of the DeepLesion class instance. This way all the requred transformations which must be done on the dataset are stored in the dataset object.

torch.utils.data.DataLoader:

This is an iterator class which provides features such as batching and shuffling. Another parameter which is used in the DataLoader class is 'collate_fn' which specifies how the samples will be batched. To illustrate how the 'collate_fn' parameter is used in this project let us recall the structure of a sample of the DeepLesion class. Each sample is a tuple:

'image' : torch.tensor ( this is the CT scan slice stored as a Tensor )
'targets' : A dictionary having keys 'boxes', 'masks', 'labels' and 'image_id'. 

The above tuple is for one sample. When the DataLoader uses the defaulf batch collate function it will maintain the same structure. In other words the DataLoader iterator will, during each iteration, return a tuple in which the first element is a list of 'image' structures and the second element is a dictionary. This dictionary will have the same keys as 'targets'. Therefore targets['boxes'] will return a list where each element in the list is itself a list of bounding boxes. Similarly targets['masks'] returns a list where each element in the list is itself a list of masks. Howevef the strucure of target which is requiored for the maskrcnn_resnet50_fpn is for 'targets' to be a list (not a dictionary) where each elements of this list is a dictionary with keys 'boxes', 'masks', 'labels'. To make this conversion a custom batch collate function is unsed (BatchCollator).

## Model:
The model employed to detect lesions when given an image of a CT scan is a Mask R-CNN with a ResNet-50-FPN backbone. A Mask R-CNN is used to detect both bounding boxes around objects (Object Detection) as well as mask segmentation (Semantic Segmentation) for each lesion object. This means that for each detected lesion a box surrounding the object is given as well as each pixel of the image is classified as being a background pixel or a lesion pixel. These two tasks combined together are called Object Instance Segmentation.

Here are the main components of the Mask R-CNN which consists of a feature extracter (backbone) followed by a Region Proposal Network and two network heads (box and mask) that run parallel to each other:

Backbone:

The ResNet50 backbone is a an architecture which acts as a feature extractor which means it takes the input image and outputs a feature map. The early layers detect low level features (edges and corners), and later layers successively detect higher level features (cars, balls, cats). The ResNet-50-FPN backbone is an improvement of this concept by using a Feature Pyramid Network. Essentially the FPN has a bottom-top pathway (which in this case is the ResNet-50) where the image is passed through a series of CNNs with down sampling (by doubling the stride at each stage). At each stage the image's spatial dimension (i.e. image resolution which is not to be confused with feature resolution) decreases however the semantic value (feature resolution) increases. This is then followed by the top-bottom pathway which takes the high level features from the last layer of the bottom-top pathway and passes them down to lower layers (using upsampling by a factor of 2 at each layer). The feaure from the top-bottom pathway at each layer are then fused with the features of the same spatial size from the bottom-up pathway via lateral connections.


Region Proposal Network (RPN):

The FPN is essentially a feature detector and not an object detector on its own. The features from the FPN are fed into a learned RPN. The RPN learns to propose regions of interest (RoI) from the image feature maps using anchors which are a set of boxes which scale according to the input image. These RoIs are regions which may contain an object.

RoIAlign:

Fast R-CNN and Faster R-CNN can be considered predecessors of Mask R-CNN. In both Fast/Faster R-CNN RoIPooling is used to extract small feature maps from each RoI window coming out of the RPN (by using max or arverage pooling). It should be noted that quantizing is performed in Fast/Faster R-CNN both going from the input image to the input image feature map where the RoI windows are projected on and also from the projected RoI windows to the small RoI feature maps created by RoIPooling. To understand quantizing take as an example if the input image is 800x800 with an RoI window of 665x665 and that the backbone CNN reduces the image and RoI by a factor of 32. In this case the dimention of the RoI window projected on the input image feature map outputed by the backbone would be floor(665/32)=20. This is called quantizing a floating point number and causes information loss. Quantizing is also done during RoIPooling when going from the projection of the RoI window on the image feature map to the RoI feature map. Quantizing in Fast/Faster R-CNN does not have a major impact as the task at hand is classification. However it has a significant negative effect on predicting pixel masks which is one of the tasks carried out in a Mask R-CNN. To account for negative effects of quantizing, a Mask R-CNN uses an RoIAlign layer which removes the quantization of RoIPool.

Box Head and Fully Connected Layers:

The RoIAligned proposals from the RPN layer are reshaped and then passed through two Fully Connected Layers to generate ROI vectors.The ROI vectors are passed through a predictor, containing 2 branches, each with an Fully Connected layer. One branch predics the object class the other is a bounding box regressor which predicts the coordinates of the bounding box. 

Mask Head and Fully Convolutional Networks:

Here the RoIAligned proposals are not reshaped as was the case in the Box Head because reshaping loses the spatial structure information necessary to generate masks. Instead the propsals are passed through a series of 3x3 convolutional layers followed by ReLU activations and 2x2 deconvolutions with stride 2 and finally a 1x1 convolution. Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers. This entire process generates the mask predictions per class.

A Mask R-CNN is basically a Faster R-CNN with an additional mask prediction branch. Two other differentiating factors are that a Mask R-CNN uses a Feature Pyramid Network and RoIAlign instead of RoIPool.

Mutiltask Loss:

The MaskRCNN class used in the implementation of this project outputs 5 loss functions. The optimization defined during training uses the sum of these 5 losses.

Classification: This log softmax score represents the error of the model correctly classifying a class object (in this case a lesion class from a non-lesion/background class)

Box Regression: Smooth L1 loss as defined in Fast R-CNN by Ross Girshick. This is a linear loss unless the absolute element-wise error falls below 1 in which case the loss is squared. It represents the error in predicting the bounding box coordinates.

Mask Loss: Binary cross entropy loss representing the error in predicting the mask segmentations.

RPN Box Regression Loss: L1 loss showing how well the proposals coming out of the RPN are. In this case a good proposal would be one in which a lesion in contained in the proposed region predicted by the RPN.  

Objectness Loss: Binary cross entropy loss. The RPN outputs many proposals however each proposal has an objectness score which represents if the anchor contains a background or foreground object. In this case a lesion is the only foreground object. The Objectness loss represents the error of the objectness score.

## Implementation and results:

The model was trained using a Mask R-CNN which was not pretrained and the network parameters are initialized using Kaiming Initialization. However in this project the ResNet50_fpn backbone used in the MaskRCNN is pretained on ImageNet. The model is run over 10 epocs. During training a loss value is output however during evaluation the model metrics are outputed. These metrics are commonly used for lesion detection and are lesion localization fraction (LLF) and non-lesion localization fraction (NLF). LLF is the total number of lesions detected (at a given threshold) divided by the total number of lesions. NLF is the total number of detected non-lesions (i.e. false positives) divided by the total number of images. During each iteration the model is trained on the training set in training mode and is then put on evaluation mode and does inference on the validation set. The best model outputs an LLF of 67.46% and NLF of 4.97.

Stochastic Gradient Descent Optimizer was used with:

Initial Learning Rate:0.001, momentum=0.9, weight_decay=0.0001

The following hyperparameters of the Mask R-CNN were also used:

rpn_pre_nms_top_n_train=12000

rpn_pre_nms_top_n_test=6000

rpn_post_nms_top_n_train=2000

rpn_post_nms_top_n_test=300

rpn_fg_iou_thresh=0.5

rpn_bg_iou_thresh=0.3

rpn_positive_fraction=0.7

bbox_reg_weights=(1.0, 1.0, 1.0, 1.0)

box_batch_size_per_image=32

In addition an anchor_generator (with scales 16, 24, 32, 48, 96 and aspect ratios 0.5, 1.0, 2.0) and an RPNHead was created and passed during model creation.

# Sanity Test

In order to quickly test the model's output a few samples are taken from the 'test' dataset. Predictions where the model has less than a 65.5% confidence score is ignored. The effect of this is that less false positive predictions are displayed. The trade-off is that some true positive results may be left out. The script for the test is contained in 'sanity_test.py'. In the below images the red bounding box is the prediction and the green bounding box is the ground truth. Predicted masks in red color have no overlap with ground truth (false positives) and ground truth masks are in blue. Where the prediction and ground truth masks overlap the color becomes pink. The confidence scores of each prediction is written on top of each red box in red.

![004409_01_01_008.png_pred.jpg](simple_test/004409_01_01_008.png_pred.jpg)
![000016_01_01_008](simple_test/000016_01_01_008.png_pred.jpg)
![000016_01_01_025.png_pred.jpg](simple_test/000016_01_01_025.png_pred.jpg)
![000016_01_01_030.png_pred.jpg](simple_test/000016_01_01_030.png_pred.jpg)

