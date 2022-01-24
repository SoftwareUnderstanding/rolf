# object-detection_yolov4
YOLOV4(you only look at once) it is the fourt version 

First we see what's new in yolov4

 YOLOV4's backbone arcitecture can be vgg16,resnet-50,resnet16-101,darknet53.... as said in official paper it is better to use "CSPdarknet53" you can check the architecture the flow chart is shown in "yolov4_model.pdf" and the code is shown in "yolov4_model.py"
 
 "cspdarknet53" is a novel backbone that can enhance the learning capability of cnn 
 
 You can download offical paper from this link https://arxiv.org/abs/2004.10934 and go through the paper to unsterstood better
 
 The neck part of yolov4 will be fpn,spp,panet..... if we use "SPP(spatial pyramid pooling)" it gives more accuracy the spp block is added over "cspdarknet53" to increase the receptive field and seperate out most signitficant features 
 
 YOLOV4 is twice as fast as efficiendet with comparable performance and fps increased by 10% to 12% compared to YOLOV3
 
 Higher input network size (resolution) – for detecting multiple small-sized objects ,More layers – for a higher receptive field to cover the increased size of input network , More parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image
 
 I have used "412 x 412" as input image shape for model
 
 "convert.py" is usedto convert the outputs of official "yolov4.weights" to "yolov4.h5" becuase load_model can recognize ".h5" or ".hdf5" format but the official weights are in ".weights" format if you didn't understood how the "convert.py" works check this blog https://medium.com/@ravindrareddysiddam/how-to-convert-yolov4-from-weights-to-h5-format-b50b244b3298 you will get an idea the blog was written by me it my or may not be a good blog but you will get an idea if you read that whole convert.py is explained in that blog
 
 "model.txt" consist all labels  as it is a cocodataset it contains 0 classes or labels 
 
 If you want to download weights the link in "weights.txt" file you can check that otherwise if you want to train on your own coustom dataset you can use man losses like ciou losses giou loss if you want reference you can check it in "losses.py" file
 
 "yolov4_.py" is just like a dense prediction contains "NMS(non max supression)" ,IOU(intersection over union)" and many functions used for images and for videos we can use "yolov4_video.py" 
 
 Finally "final.py" is used for combining all theses files and to get the output for images whereas for videos we can use "video.ipynb" 
 
 
## ABOUT YOLOV4
***YOLOV4 [you only look at once version 4]***

Architecture : input ->Backbone -> Neck -> Dense prediction -> sparse prediction

Input: Image, Patches, Image Pyramid

Backbones: VGG16 , ResNet-50 , SpineNet, EfficientNet-B0/B7 ,CSPResNeXt50,CSPDarknet53

Neck:
       Additional blocks: SPP , ASPP , RFB, SAM 
       Path-aggregation blocks: FPN, PAN ,NAS-FPN, Fully-connected FPN, BiFPN, ASFF , SFAM 

Heads:
       Dense Prediction (one-stage):◦ RPN , SSD, YOLO , RetinaNet (anchor based)◦ CornerNet, CenterNet, MatrixNet, FCOS  (anchor free)
       Sparse Prediction (two-stage):◦ Faster R-CNN , R-FCN, Mask RCNN (anchor based)◦ RepPoints(anchor free)


***Mainly two types of models in object detection :***

  One stage or two stage models. A one stage model is capable of detecting objects without the need for a preliminary step. On the contrary, a two stage detector uses a preliminary stage where regions of importance are detected and then classified to see if an object has been detected in these areas. The advantage of a one stage detector is the speed it is able to make predictions quickly allowing a real time use.
  
   A modern detector is usually composed of two parts,a backbone which is pre-trained on ImageNet and a head which is used to predict classes and bounding boxes of objects. For those detectors running on GPU platform, their backbone could be VGG , ResNet , ResNeXt,or DenseNet. For those detectors running on CPU platform, their backbone could be SqueezeNet, MobileNet, or ShuffleNet. As to the head part,it is usually categorized into two kinds, i.e., one-stage object detector and two-stage object detector. The most representative two-stage object detector is the R-CNN series,including fast R-CNN , faster R-CNN , R-FCN,and Libra R-CNN. It is also possible to make a twostage object detector an anchor-free object detector, such as RepPoints. As for one-stage object detector, the most representative models are YOLO , SSD ,and RetinaNet. In recent years,anchor-free one-stage object detectors are developed. The detectors of this sort are CenterNet, CornerNet, FCOS

***Why YoloV4 ?***

  YoloV4 is an important improvement of YoloV3, the implementation of a new architecture in the Backbone and the modifications in the Neck have improved the mAP(mean Average Precision) by 10% and the number of FPS(Frame per Second) by 12%. In addition, it has become easier to train this neural network on a single GPU.

***Backbone***

  Deep neural network composed mainly of convolution layers. The main objective of the backbone is to extract the essential features, the selection of the backbone is a key step it will improve the performance of object detection. Often pre-trained neural networks are used to train the backbone.
The YoloV4 backbone architecture is composed of three parts:

***Bag of freebies***

***Bag of specials***

***CSPDarknet53***

***Bag of freebies:***

  which can make the object detector receive better accuracy without increasing the inference cost. We call these methods that only change the training strategy or only increase the training cost as bag of freebies .

***=> Data augmentation:-***
                The main objective of data augmentation methods is to increase the variability of an image in order to improve the generalization of the model training.
   The most commonly used methods are Photometric Distortion, Geometric Distortion, MixUp, CutMix and GANs.
   
***=> Photometric distortion:-***
               Photometric distortion creates new images by adjusting brightness, hue, contrast, saturation and noise to display more varieties of the same image.
               
***=> Geometric distorsion:-***
                 The geometric distortion methods are all the techniques used to rotate the image, flipping, random scaling or cropping.

***=>MixUp:-***
        Mixup augmentation is a type of augmentation where in we form a new image through weighted linear interpolation of two existing images. We take two images and do a linear combination of them in terms of tensors of those images. Mixup reduces the memorization of corrupt labels, increases the robustness to adversarial examples, and stabilizes the training of generative adversarial networks.
        
 ***=> CutMix:-***
       CutMix augmentation strategy: patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances
       
  ***=> Focal loss:-***
       The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training . The new Focal loss function is based on the cross entropy by introducing a (1-pt)^gamma coefficient. This coefficient allows to focus the importance on the correction of misclassified examples. at gamma =0 focal loss= cross entropy
       
  ***=> Label smoothing:-***
         Whenever you feel absolutely right, you may be plainly wrong. A 100% confidence in a prediction may reveal that the model is memorizing the data instead of learning. Label smoothing adjusts the target upper bound of the prediction to a lower value say 0.9. And it will use this value instead of 1.0 in calculating the loss. This concept mitigates overfitting.
         
***=> IoU loss:-***
        Most object detection models use bounding box to predict the location of an object. To evaluate the quality of a model the L2 standard is used, to calculate the difference in position and size of the predicted bounding box and the real bounding box.
        
The disadvantage of this L2 standard is that it minimizes errors on small objects and tries to minimize errors on large bounding boxes.

To address this problem we use IoU loss for the YoloV4 model.

Compared to the l2 loss, we can see that instead of optimizing four coordinates independently, the IoU loss considers the bounding box as a unit. Thus the IoU loss could provide more accurate bounding box prediction than the l2 loss. Moreover, the definition naturally norms the IoU to [0, 1] regardless of the scales of bounding boxes
Recently, some Improved IoU loss are For example, GIoU loss  is to include the shape and orientation of object in addition to the coverage area. They proposed to find the smallest area BBox that can simultaneously cover the predicted BBox and ground truth BBox, and use this BBox as the denominator to replace the denominator originally used in IoU loss. As for DIoU loss , it additionally considers the distance of the center of an object, and CIOU loss , on the other hand simultaneously considers the overlapping area, the distance between center points, and the aspect ratio. CIoU can achieve better convergence speed and accuracy on the BBox regression problem.

 ***Bag of specials***
    Bag of special methods are the set of methods which increase inference cost by a small amount but can significantly improve the accuracy of object detection.
    
   ***=> Mish activation:-***
          Mish is a novel self-regularized non-monotic activation function which can be defined by f(x) = x tanh(softplus(x)).
   Why Mish activation :Due to the preservation of a small amount of negative information, Mish eliminated by design the preconditions necessary for the Dying ReLU phenomenon. A large negative bias can cause saturation of the ReLu function and causes the weights not to be updated during the backpropagation phase making the neurons inoperative for prediction.
   
   ***=> SPP block:-***
               SPP module was originated from Spatial Pyramid Matching (SPM) [39], and SPMs original method was to split feature map into several d × d equal blocks, where d can be{1, 2, 3, ...}, thus forming spatial pyramid, and then extracting bag-of-word features. SPP integrates SPM into CNN and use max-pooling operation instead of bag-of-word operation. Since the SPP will output one dimensional feature vector, it is infeasible to be applied in Fully Convolutional Network (FCN).
  The post-processing method commonly used in deeplearning-based object detection is NMS, which can be used to filter those BBoxes that badly predict the same object, and only retain the candidate BBoxes with higher response. The way NMS tries to improve is consistent with the method of optimizing an objective function

***CSPDarknet53***

  The Cross Stage Partial architecture is derived from the DenseNet architecture which uses the previous input and concatenates it with the current input before moving into the dense layer.
    
Each stage layer of a DenseNet contains a dense block and a transition layer, and each dense block is composed of k dense layers. The output of the ith dense layer will be concatenated with the input of the ith dense layer, and the concatenated outcome will become the input of the (i + 1)th dense layer. The equations showing the above-mentioned mechanism can be expressed as:

 x1=w1 x x0
 
 x2= w2 x [x0,x1]
 
 .
 
 .
 
 xk=wk x [x0,x1,x2,x3,......,xk-1]

where x is the convolution [x0,x1,x2 ....] concatenate of x0,x1,x2,....

wi and xi are the weights and output respetively 

The CSP is based on the same principle except that instead of concatenating the ith output with the ith input, we divided the input ith in two parts x0' and x0'’, one part will pass through the dense layer x0'’, the second part x0' will be concatenated at the end with the result at the output of the dense layer of x0'’.

***Neck (detector)***

 The essential role of the neck is to collect feature maps from different stages. Usually, a neck is composed of several bottom-up paths and several top-down paths.
SPP

What is the problem caused by CNN and fully connected network ?

The fully connected network requires a fixed size so we need to have a fixed size image, when detecting objects we don’t necessarily have fixed size images. This problem forces us to scale the images, this method can remove a part of the object we want to detect and therefore decrease the accuracy of our model.

The second problem caused by CNN is that the size of the sliding window is fixed.

***How SPP runs ?***
     At the output of the convolution neural networks, we have the features map, these are features generated by our different filters. To make it simple, we can have a filter able to detect circular geometric shapes, this filter will produce a feature map highlighting these shapes while keeping the location of the shape in the image.
Spatial Pyramid Pooling Layer will allow to generate fixed size features whatever the size of our feature maps. To generate a fixed size it will use pooling layers like Max Pooling for example, and generate different representations of our feature maps.
steps carried out by an SPP.

1)suppose we have a 3-level PPS. Suppose the conv5 (i.e. the last convolution layer) has 256 features map.

2)First, each feature map is pooled to become a one value. Then the size of the vector is (1, 256)

3)Then, each feature map is pooled to have 4 values. Then the size of the vector is (4, 256)

On the same way, each feature is pooled to have 16 values. Then the size of the vector is (16, 256)

The 3 vectors created in the previous 3 steps are then concatenated to form a fixed size vector which will be the input of the fully connected network.

***What are the benefits of SPP ?***
      SPP is able to generate a fixed- length output regardless of the input size

SPP uses multi-level spatial bins, while the sliding window pooling uses only a single window size. Multi-level pooling has been shown to be robust to object deformations

SPP can pool features extracted at variable scales thanks to the flexibility of input scales

***PaNet:*** for aggregate different backbone levels

In the early days of deep learning, simple networks were used where an input passed through a succession of layers. Each layer takes input from the previous layer. The early layers extract localized texture and pattern information to build up the semantic information needed in the later layers. However, as we progress to the right, localized information that may be needed to fine-tune the prediction may be lost.

To correct this problem, PaNet has introduced an architecture that allows better propagation of layer information from bottom to top or top to bottom.that the information of the first layer is added in layer p5, and propagated in layer N5 . This is a shortcut to propagate low level information to the top.

In the original implementation of PaNet, the current layer and information from a previous layer is added together to form a new vector. In the YoloV4 implementation, a modified version is used where the new vector is created by concatenating the input and the vector from a previous layer.

***Head (detector)***

  The role of the head in the case of a one stage detector is to perform dense prediction. The dense prediction is the final prediction which is composed of a vector containing the coordinates of the predicted bounding box (center, height, width), the confidence score of the prediction and the label.

***CIOU-loss***

The CIoU loss introduces two new concepts compared to IoU loss. The first concept is the concept of central point distance, which is the distance between the actual bounding box center point and the predicted bounding box center point.

The second concept is the aspect ratio, we compare the aspect ratio of the true bounding box and the aspect ratio of the predicted bounding box. With these 3 measures we can measure the quality of the predicted bounding box.

Lciou=1-iou+p2(b,bgt)/c2+alpha x v

where b and bgt denote the central points of B and Bgt,p(.) is the Euclidean distance, c is the diagonal length of the smallest enclosing box covering the two boxes, α is a positive trade-off parameter, and v measures the consistency of aspect ratio.

v = 4/pie2 (arctan(wgt/hgt) - artan(w/h))2

where h is the height of the bounding box and w is the width.

***CmBN (Cross mini Batch Normalization)***

  Why use Cross mini Batch Normalization instead of Batch Normalization? What are its advantages and how does it work?

Batch Normalization does not perform when the batch size becomes small. The estimate of the standard deviation and mean is biased by the sample size. The smaller the sample size, the more likely it is not to represent the completeness of the distribution. To solve this problem, Cross mini Batch Normalization is used, which uses estimates from recent batches to improve the quality of each batch’s estimate. A challenge of computing statistics over multiple iterations is that the network activations from different iterations are not comparable to each other due to changes in network weights.
 
 Batch normalization =xt,i(theta t)- mue t (theta t)/root(sigma(theta t)2+ epsilon)

where ε is a small constant added for numerical stability, and μt(θt) and σt(θt) are the mean and variance computed for all the examples from the current mini-batch.

whereas in Cross mini Batch Normalization the mean and variance are calculated from the previous N means and variances and approximated using Taylor formulae to express them as a function of the parameters θt rather than θt-N.

***DropBlock regularization***

   Neural networks work better if they are able to generalize better, to do this we use regularization techniques such as dropout which consists in deactivating certain neurons during training. These methods generally improve accuracy during the test phase.

Nevertheless the dropout drops features randomly, this method works well for fully connected layers but is not efficient for convoluted layers where features are spatially correlated.

In DropBlock, features in a block (i.e. a contiguous region of a feature map), are dropped together. As DropBlock discards features in a correlated area, the networks must look elsewhere for evidence to fit the data.

***Mosaic data augmentation***

  Mosaic data augmentation combines 4 training images into one in certain ratios. This allows for the model to learn how to identify objects at a smaller scale than normal. It also encourages the model to localize different types of images in different portions of the frame.

***Self-Adversarial Training (SAT)***

  Self-Adversarial Training (SAT) represents a new data augmentation technique that operates in 2 forward backward stages. In the 1st stage the neural network alters the original image instead of the network weights. In this way the neural network executes an adversarial attack on itself, altering the original image to create the deception that there is no desired object on the image. In the 2nd stage, the neural network is trained to detect an object on this modified image in the normal way with original label before add noise to the image.

***Eliminate grid sensitivity***

  Eliminate grid sensitivity the equation bx = σ(tx)+ cx,by =σ(ty)+cy, where cx and cy a real ways whole numbers, is used in YOLOv3 for evaluating the object coordinates, therefore, extremely high tx absolute values are required for the bx value approaching the cx or cx + 1 values. We solve this problem through multiplying the sigmoid by a factor exceeding 1.0, so eliminating the effect of grid on which the object is undetectable.

Using multiple anchors for a single ground truth

  We predict several boxes, because it is difficult for a convolution network to predict directly a set of boxes associated with objects of different ratio, that’s why we use anchors that divide the image space according to different strategies.
  
From the features map created by the convolution layers, we create many anchor boxes of different ratios in order to be able to represent objects of any size, we then decide thanks to the IOU to assign some boxes to an object or a background according to the threshold below.
IoU (truth, anchor) > IoU threshold (formula)

***Cosine annealing scheduler***

A cosine function is used to update the learning rate, the advantage of the cosine function is that it is cyclic allowing to get out of the local minima more easily than the step method or SGD.

***Optimal hyper- parameters***

To try to find the best hyperparameters, genetic algorithms are used to find the most suitable parameters. N randomly selected parameters are initialized. Then we train N models, select the K best models, then we choose random parameters derived from the K best models and we train N2 new models and we start again until we reach the final iteration.

Random training shapes

Many single-stage object detectors are trained with a fixed input image shape. To improve generalization, we can train the model with different image sizes. (Multi-Scale Training in YOLO)

***SAM block***

   SAM simply consists of applying two separate transforms to the output feature map of a convolutional layer, a Max Pooling and an Avg Pooling. The two features are concatenated and then passed in a convoluted layer, before applying a sigmoid function that will highlight where the most important features are located.

***DIoU-NMS***

NMS (Non-Maximum Suppression) is used to remove the boxes that represent the same object while keeping the one that is the most precise compared to the real bounding box.

R diou = p2(b,bgt)/c2

where b and bgt denote the central points of B and Bgt, ρ(·) is the Euclidean distance, and c is the diagonal length of the smallest enclosing box covering the two boxes.

si={  si ,iou - R diou <e,
   
   {  0,iou - R diou >e

We test if the overlap rate minus the distance between the two centers is lower than the threshold ε, if this is the case we keep the bounding box, otherwise we delete


The center position of the bounding box in the image (bx, by)

The width of the box( bw )

The height of the box ( bh )

The class of object ( c )

y=(pc,bx,by,bh,bw,c)
