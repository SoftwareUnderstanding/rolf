# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4).

## The Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Proposed alternate solution

Computer vision techniques like the ones proposed above were widely used for classification and segmentation in the past, but the advances in deep learning architectures have made many of these approaches almost obsolete.

Since 2012, the winner of classification and object detection competitions (ImageNet, CIFAR, MSCOCO, etc) have always been Convolutional Neural Networks - this is a strong indication that these architectures are much better suited for the ultimate goal of this project - detecting vehicles on a video stream.

Instead of following the suggested steps above, you will find below an alternate path, that will discuss how to implement the current state-of-the-art architecture for object detection.

* Explore the dataset.
* Find a suitable test/validation split.
* Propose a deep learning architecture.
* Train a model from scratch and discuss results.
* Compare both results.
* Run the pipeline on a video stream.

## Dataset exploration

We are going to use the [Udacity labeled dataset 1](https://github.com/udacity/self-driving-car/tree/master/annotations).

>The dataset includes driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across 9,423 frames collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans.

The dataset are frames from a video of roughly 80 minutes of continuous driving in California and if we are not careful about how we split our data there will be a lot of information leaking from the training set into the validation set and our results will not be representative of the real performance of the pipeline.

## Splitting the dataset

After a meticulous analysis of all provided frames, I have selected 998 frames to be our validation set. That would be equivalent of removing a little over 8 minutes of driving from our training data set.

The `truck` and `car` classes were combined into a new `vehicle` class and the bounding boxes for the `pedestrian` class were ignored.

## Data pre-processing

During the visual exploration of the dataset it also became evident that many frames are still too similar to each other, even with the video being recorder at 2Hz.

To adress this issue while also being mindful that we will need a large dataset in order to properly train our model, I came up with the following pre-processing steps:

* Skip every other frame to resample our training data to 1Hz.
* Create an image mask for each frame we are going to use (this will make things a lot easier during data augmentation).
* Augment our dataset using random rotations, scalings, translations and crops.

Now that we have image masks with the same dimensions of the input image, we can define a data augmentation pipeline. We apply the same transformations to both the input image and the image mask, ensuring we will always have the exact location of the the vehicles highlighted in the augmented dataset as well.

For the data augmentation parameters we will leverage from knowledge acquired while implementing our [Traffic Sign Classifier](https://github.com/bguisard/CarND-Traffic-Sign-Classifier-Project).

- Rotation between -15 and 15 degrees
- Scaling between 0.9 and 1.1 times
- Translation between -5% and +5% of image width
- Shear limit between -10 and 10 degrees
- Motion blur kernel size of 3 pixels

On top of those transformations we will also reduce the input size to cut down on computation time. This will be done in two different steps, first scaling the images down to 20% of its original size and randomly cropping the image to 224 x 224 - the same size as ImageNet images, so we can leverage from pretrained DenseNet weights if we want.

The validation set will not receive any augmentation, but will need to be scaled down to the same size as our training dataset. To keep the objects with the same proportion as they are in a live video stream, we will scale down the lower dimension of the image to 224 pixels and do a center crop to fit the other dimension to 224 rather than scaling it down.

You can find below two examples of each case.

### Augmented training sample

![alt text][image6]

![alt text][image7]

### Cropped validation sample

![alt text][image8]

![alt text][image9]

## Densely Connected Convolutional Networks

Also known as `DenseNets`[1] is a fairly new architecture, published in late 2016, that expands ideas introduced by `ResNets`[2], where blocks of convolutional layers receives not only the feature-maps of the previous block, but also it's input as well. Since they are forward feeding not only their outputs, but also their inputs, the optimization process is done on the residuals of each transformation, hence the name Residual Networks.

The biggest changes introduced by DenseNets are that the feature-maps of each block are passed to all subsequent layers of each block and that these feature-maps are concatenated together instead of summed.

>For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling
advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR- 10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less memory and computation to achieve high performance.

![alt text][image1]

source: https://arxiv.org/pdf/1608.06993v3.pdf

## The one hundred layers Tiramisu

Following the great results shown by DenseNets in image classification, Jégou et al. [3] extended the previous work, proposing an architecture for semantic image segmentation that uses several dense blocks during downsampling and upsampling.

The proposed network is called Fully Convolutional DenseNets for Semantic Segmentation, but was also named "The One Hundred Layers Tiramisu" and the diagram below gives a great overview of the architecture.

![alt text][image2]
>Diagram of our architecture for semantic segmentation. Our architecture is built from dense blocks. The diagram is com- posed of a downsampling path with 2 Transitions Down (TD) and an upsampling path with 2 Transitions Up (TU). A circle repre- sents concatenation and arrows represent connectivity patterns in the network. Gray horizontal arrows represent skip connections, the feature maps from the downsampling path are concatenated with the corresponding feature maps in the upsampling path. Note that the connectivity pattern in the upsampling and the downsam- pling paths are different. In the downsampling path, the input to a dense block is concatenated with its output, leading to a linear growth of the number of feature maps, whereas in the upsampling path, it is not.

source:https://arxiv.org/pdf/1611.09326v2.pdf

The main architecture described on their work is a 103 layer network, called FC-DenseNet103. The definition of each of the key blocks of the model can be seen below.

>Dense block layers are composed of BN, followed by ReLU, a 3 × 3 same convolution (no resolution loss) and dropout with probability p = 0.2. The growth rate of the layer is set to k = 16.

>Transition down is composed of BN, followed by ReLU, a 1 × 1 convolution, dropout with p = 0.2 and a non-overlapping max pooling of size 2 × 2.

>Transition up is composed of a 3×3 transposed convolution with stride 2 to compensate for the pooling operation.

And the overview of the architecture:

| Architecture                 |
|:----------------------------:|
|Input, m = 3                  |
|3×3 Convolution, m = 48       |
|DB (4 layers) + TD, m = 112   |
|DB (5 layers) + TD, m = 192   |
|DB (7 layers) + TD, m = 304   |
|DB (10 layers) + TD, m = 464  |
|DB (12 layers) + TD, m = 656  |
|DB (15 layers), m = 880       |
|TU + DB (12 layers), m = 1072 |
|TU + DB (10 layers), m = 800  |
|TU + DB (7 layers), m = 560   |
|TU + DB (5 layers), m = 368   |
|TU + DB (4 layers), m = 256   |
|1×1 Convolution, m = c        |
|Softmax                       |

>Architecture details of FC-DenseNet103 model used in our experiments. This model is built from 103 convolutional layers. In the Table we use following notations: DB stands for Dense Block, TD stands for Transition Down, TU stands for Transition Up, BN stands for Batch Normalization and m corresponds to the total number of feature maps at the end of a block. c stands for the number of classes.

source:https://arxiv.org/pdf/1611.09326v2.pdf

## The model

The power of very deep DenseNets was already proven by the authors of [1, 3], so the focus of this work will be in testing the performance of smaller versions of these networks to vehicle detection.

They are an excellent choice for autonomous vehicles as they can deliver state-of-the-art performance while using only a fraction of the parameters - usually about 10 fold reduction when compared to other similar performing architectures.

My goal is not to just deliver a pipeline that accurately identifies vehicles on a video stream, but I want to do so with a small footprint (for scalability) and real-time performance.

The starting point was the 56 layer FC-DenseNet56 as proposed in [3], but some minor changes were made. The table below shows our final architecture.

| Architecture                 |
|:----------------------------:|
|Input, m = 3                  |
|3×3 Convolution, m = 24       |
|DB (4 layers) + TD, m = 72    |
|DB (4 layers) + TD, m = 120   |
|DB (4 layers) + TD, m = 168   |
|DB (4 layers) + TD, m = 216   |
|DB (4 layers) + TD, m = 264   |
|DB (4 layers), m = 312        |
|TU + DB (4 layers), m = 360   |
|TU + DB (4 layers), m = 312   |
|TU + DB (4 layers), m = 264   |
|TU + DB (4 layers), m = 216   |
|TU + DB (4 layers), m = 168   |
|1×1 Convolution, m = 1        |
|Sigmoid                       |

## Training

I leveraged from a Python generator and our preprocessing function to create a virtually unlimited training dataset. Differently than what was found by the authors of [3], I've found that Adam was a much more efficient optimizer for our problem than RMSProp.

Batch size was set at 4, limited by the memory of the GPU, and it was first optimized for 20 epochs with Adam and learning rate 1e-3 and then for another 20 epochs with learning rate 1e-4. The recommended dropout of 0.2 and weight decay of 1e-4 were both used to prevent overfitting.

After 40 epochs the training loss was significantly lower than the validation loss, which is usually a sign that the augmentation is too strong and at that point augmentation was turned off before training the model for another 20 epochs.

## Validation results

Jégou et al. reported a mean IoU (intersection over union) accuracy of 73.2% for the car class, while using the FC-DenseNet56 architecture, but the overall model accuracy was 88.9%. It took us 60 epochs (roughly 10 hours) to get to 72.3% in a different dataset, but this model was  trained to identify just one class, so there was clearly room for improvement.

During the first attempts to optimize the model, the suggested IoU loss was not efficient and the model wouldn't start converging, so I used a weighted binary cross entropy loss. After 10h of optimization, however, the model had a rough idea of where the vehicles were, so I thought that this time IoU could improve our accuracy and decided to try it again, with the goal of getting a similar performance than the authors of [3].

After another 40 epochs (and another 8 hours), the accuracy was much better, with mean IoU constantly above 86%, so I decided to stop training, but it is possible that it would continue to improve if given more time to train.

### Examples (before and after retraining with IoU)
#### 1)
![alt text][image3]

![alt text][image10]

#### 2)
![alt text][image4]

![alt text][image11]

#### 3)
![alt text][image5]

![alt text][image12]

## Results on video stream

With a very strong performance on the validation set, the model was tested on the provided video and the results were very promising. The model was able to detect the location of vehicles accurately in all lanes, but it encountered some difficulties in scenarios that were not in the training set (e.g change in light conditions and trash on the side of the highway).

To address those minor details I curated a fine tuning set of a little under 50 frames and retrained the model with a very low learning rate (1e-5) over 10 epochs.

The results this time were very encouraging and can be seen in this [video](./video_output/project_video.mp4).

## Final considerations

In this report I demonstrated that FC-DenseNets are a great alternative to conventional computer vision methods of vehicle detection. The 56 layer network that was proposed has over 86.8% mean IoU with only 1.1 million parameters and the weights file has only 5MB. This makes it viable not only for self-driving cars, but also for other autonomous vehicles with restricted computing power, like drones.

The final model performed really well on a video stream that was completely new to it. Although it did require a very small fine tuning, this can easily be prevented in future implementations by curating a better training dataset. It's good enough out-of-the-box that it can also be used to generate training data for its own fine tuning.


## References
[1] [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v3.pdf)

[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

[3] [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v2.pdf)

[4] [Dense Net in Keras](https://github.com/titu1994/DenseNet/)

[5] [Small U-Net for vehicle detection](https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6)

[//]: # (Image References)

[image1]: ./images/dense_block.jpg "A dense block example"
[image2]: ./images/tiramisu.jpg "Fully Convolutional DenseNets for Semantic Segmentation"
[image3]: ./images/test_run_1.png "Before training with IoU"
[image4]: ./images/test_run_2.png "Before training with IoU"
[image5]: ./images/test_run_3.png "Before training with IoU"
[image6]: ./images/augmented_train1.png "Augmented training sample"
[image7]: ./images/augmented_train2.png "Augmented training sample"  
[image8]: ./images/processed_val1.png "Validation sample"
[image9]: ./images/processed_val2.png "Validation sample"
[image10]: ./images/test_newrun_1.png "After training with IoU"
[image11]: ./images/test_newrun_2.png "After training with IoU"
[image12]: ./images/test_newrun_3.png "After training with IoU"
