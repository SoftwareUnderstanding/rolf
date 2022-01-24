[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project - Follow Me ##

In this project we will train a deep neural network, especially Fully Convolutional Neural Network (FCN) to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/followme.jpg
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

We used above data training and validation for train weight for FCN.

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

# Software used for Project Training:

* Windows 8.1 64bit
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

# Hardware used for Project Training:

* Notebook ASUS N56V
- Intel(R) Core(TM) i7 - 3630QM
- RAM 8 GB
* NVIDIA GeForce 650M (2 GB with 384 core)

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Write Up Report Follow Me Project
## Write Up by Dedi

# Networks
### Convolutional Neural Network (CNN)
Convolutional neural network have architecture as image below:
<p align="center"> <img src="./docs/misc/CNN_Architecture.png"> </p>
<p align="center"> <img src="./docs/misc/CNN_Classifications.png"> </p>

A Convolution Neural Network may have several layer which is each layer might capture a different level in the hierarchy of object. The first layer called as lowest level hierarchy, where CNN may classifies small parth of the image into simple shapes like horizontal and vertical linea and simple blobs of colors. The last layers tend to be highest level in the hierarchy and may classify more complex ideas like shapes and eventually full object like cars. All this convolutional layer also called as feature learning.
  
The highest level of hierarchy or the last convolutional layer then connected with classification layer which is consist with fully connected layer and softmax . From this layer, the input would be classified as which object. CNN are usually used to classify object inside an image.

### Fully Convolutional Neural Network (FCN)
CNN very useful for tackling tasks such as image classification, which just want to determine 'what' is the object in a image. But when we want to know 'where' is in the image a certain object, CNN would not work since fully connected layers remove any sense of spacial information. Therefore Fully Convolutional Network (FCN), will perform this task.

A FCN is a CNN, which is the classification layer is replace with 1x1 convolution layer with a large "receptive field" and add with upscale layer which called as decoder.  The purpose in here is to get the global context of the scene and enable us to get what are the object on image and their spatial information. The output of this network not only contain object classification but also the scene of segmentation.

<p align="center"> <img src="./docs/misc/FCN.png"> </p>

The structure of FCN is divide by two part that is encoder layer part which will extract feature from the image and decoder layer part which will upscale the output of the encoder so the output will have the original size of the image. This two part connected with 1x1 convolution layer.

a 1x1 convolution simply maps an input pixel with all its channel to an output pixel, not looking at anything around itself. It is often used to reduce the number of depth channels, since it is often very slow to multiply volumes with extremely large depths.

When we convert our last fully connected (FC) layer of the CNN to a 1x1 convolutional layer we choose our new conv layer to be big enough so that it will enable us to have this localization effect scaled up to our original input image size then activate pixels to indicate objects and their approximate locations in the scene as shown in above figure. replacement of fully-connected layers with convolutional layers presents an added advantage that during inference (testing your model), you can feed images of any size into your trained network.

In GoogLeNet architecture, 1x1 convolution is used for two purposes
* To make network deep by adding an "inception module" like Network in network paper
* To reduce the dimensions inside this "inception module"

Here is the screenshot from the paper, which elucidates above points:
<p align="center"> <img src="./docs/misc/inception_1x1.png"> </p>

It can be seen from the image on the right, that 1x1 convolutions (in yellow), are specially used before 3x3 and 5x5 convolution to reduce the dimensions. It should be noted that a two step convolution operation can always to combined into one, but in this case and in most other deep learning networks, convolutions are followed by non-linear activation and hence convolutions are no longer linear operators and cannot be combined.

Image without Skip Connections:
<p align="center"> <img src="./docs/misc/FirstResultFCN_No_Skips.png"> </p>

Everytime we do convolution (down sampling), we are facing one problem with this approach that is we lose some information; we keep the smaller picture (the local context) and lose the bigger picture (the global context) for example if we are using max-pooling to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.

To solve this problem we also get some activation from previous layers and sum/interpolate them together. This process is called "skip" from the creators of this algorithm.

Those up-sampling operations used on skip are also learn-able.

<p align="center"> <img src="./docs/misc/Skip_Layers_FCN.png"> </p>

Below we show the effects of this "skip" process, notice how the resolution of the segmentation improves after some "skips"

<p align="center"> <img src="./docs/misc/AllSkips_FCN.png"> </p>

# Data Collection

In this learning project, I didn't record train, validation and sample_evaluation_data data from quadcopter simulator. I used train and validation data from link above to get weight from the network model that I have design. 

<table><tbody>
    <tr><th align="center" colspan="3"> Data Set 1</td></tr>
    <tr><th align="center">Folder</th><th align="center">Content</th></tr>
    <tr><td align="left">/data/train</td><td align="left">4,131 images + 4,131 masks</td></tr>
    <tr><td align="left">/data/validation</td><td align="left">1,184 images + 1,184 masks</td></tr>    
    <tr><td align="left">/data/sample_evalution_data/following_images</td>
       <td align="left">542 images + 542 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_non_targ</td>
       <td align="left"> 270 images + 270 masks</td></tr><tr>
    <td align="left">/data/sample_evalution_data/patrol_with_targ</td>
       <td align="left"> 322 images + 322 masks</td></tr>
</tbody></table>

# FCN Layers

In this project, there are seven layers to build a fully convolutional networks (FCN). Three layers for encoder, one layers as one by one convolutional matrix and another three layers as decoder block. See image below:

<p align="center"> <img src="./docs/misc/FCN Design.jpg"> </p>

* The first layer is encoder block layer which have input from image. This layer have filter width 32 and 2 strides.
* The second layer is encoder block layer which have input from first layer. This layer have filter width 64 and 2 strides.
* The third layer is encoder block layer which have input from second layer. This layer have filter with width 128 and build from 2 strides.
* The fourth layer is 1x1 convolution layer using convolution 2D batch normalize. This layer build with filter width 256, and kernel 1x1 with 1 strides.
* The fifth layer is decoder block layer which have input from 1x1 convolution layer and skip connection from second layer.
* The sixth layer is decoder block layer which have input from fifth layer and skip connection from the first layer. The dimension of this layer is same with first layer.
* The last layer is decoder block layer which have input from sixth layer and skip connection from input image. The last layer is an output layer of FCN. This layer have the dimension as same as input image.

Explanations of how to build the code for FCN Design above would be explain in Build the Model section below

# Build The Model

### Separable convolution layer:
The Encoder for FCN require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for your use. Each includes batch normalization with the ReLU activation function applied to the layers.
```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

### Bilinear Upsampling
The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended, but you can try out different factors as well. Upsampling is used in the decoder block of the FCN.
```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

## TODO Code

### Encoder Block
Create an encoder block that includes a separable convolution layer using the separable_conv2d_batchnorm() function. The filters parameter defines the size or depth of the output layer.
```python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

### Decoder Block
The decoder block is comprised of three parts:
*    A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
*    A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
*    Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample_small_ip_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([upsample_small_ip_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    output_layer = separable_conv2d_batchnorm( output_layer, filters, strides=1)
    
    return output_layer
```

### The FCN Model
Now that you have the encoder and decoder blocks ready, go ahead and build your FCN architecture!

There are three steps:

*    Add encoder blocks to build the encoder layers. This is similar to how you added regular convolutional layers in your CNN lab.
*    Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
*    Add decoder blocks for the decoder layers.

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    layer01 = encoder_block(inputs , filters=32 , strides=2)
    
    layer02 = encoder_block(layer01, filters=64 , strides=2)
    
    layer03 = encoder_block(layer02, filters=128, strides=2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    layer04 = conv2d_batchnorm(layer03, filters=256, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    layer05 = decoder_block(layer04, layer02, filters=128 )
    
    layer06 = decoder_block(layer05, layer01, filters=64  )
    
    layer07 = decoder_block(layer06, inputs , filters=32  )
    
    # The function returns the output layer of your model. "layer07" is the final layer obtained from the last decoder_block()
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer07)
    print("Outputs shape:",outputs.shape, "\tOutput Size in Pixel")
    
    return outputs
```

#Training
Training is my bigest problems. I have facing two problem in AWS account, the first one is AWS reject my request increasing EC2 instance p2.xlarge and the second ones is AWS facing problem when send my promotion code for initial balance by mail. Thanks for support dashboard in AWS, my complain had been approved at 18 June for initial credit. And at 20 June AWS approved for increasing limit in p2.xlarge when I reopen the case. Sadly when all my request had been approved, I moved to my village with lower internet connectivity speed therefore I used my laptop for trained the model. Spesification of my laptop you can see at above explanations.

To increase the speed of training the model in my laptop, I install the following software and library:
* Latest NVIDIA Driver 398.11
* CUDA v9.0
* CuDNN v7.1

## Hyperparameters

### Batch Size
Number of training samples/images that get propagated through the network in a single pass. In this training we used batch_size with value 32.
```python
learning_rate = 0.001
batch_size = 32
```
Learning rate number I had selected is 0.001. I select this lowest number because the first one I select is 0.05 with failure final score 33%.

### Epochs
#### Number of Epochs
Number of times the entire training dataset gets propagated through the network. In this training we choose the total number of epochs were 20.

#### Step Each Epoch
Number of batches of training images that go through the network in 1 epoch. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size. Total number in training data set is 4131 images divided by 32 with the result is 129. We select step each epoch is 200.

```python
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

This training is very hard because I need almost 24 hour to get my model training result get finished. Below is my plotting training loss and validation loss using 20 epochs.

<p align="center"> <img src="./docs/misc/Training.png"> </p>

Each epoch required 4114 second, therefore 20 epochs would be need about 82280 second or 22.8 hour. Detail graphics you can see in model_training.html

#Prediction
Now that you have your model trained and saved, you can make predictions on your validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well your model is doing under different conditions.

There are three different predictions available from the helper code provided:

*    patrol_with_targ: Test how well the network can detect the hero from a distance.
*    patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
*    following_images: Test how well the network can identify the target while following them.

## Patrol with Target
<p align="center"> <img src="./docs/misc/pwt1.png"> </p>
<p align="center"> <img src="./docs/misc/pwt2.png"> </p>
<p align="center"> <img src="./docs/misc/pwt3.png"> </p>

## Patrol without Target
<p align="center"> <img src="./docs/misc/pwot1.png"> </p>
<p align="center"> <img src="./docs/misc/pwot2.png"> </p>
<p align="center"> <img src="./docs/misc/pwot3.png"> </p>

## Patrol with Target while following them
<p align="center"> <img src="./docs/misc/fi1.png"> </p>
<p align="center"> <img src="./docs/misc/fi2.png"> </p>
<p align="center"> <img src="./docs/misc/fi3.png"> </p>

#Evaluation
Evaluate our model! The following cells include several different scores to help you evaluate your model under the different conditions discussed during the Prediction step.
### Scores for while the quad is following behind the target.
```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9944914007764788
average intersection over union for other people is 0.3256942366738677
average intersection over union for the hero is 0.9125996469040777
number true positives: 539, number false positives: 0, number false negatives: 0
```
### Scores for images while the quad is on patrol and the target is not visable
```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.981193497537517
average intersection over union for other people is 0.6976223997700709
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 52, number false negatives: 0
```
### This score measures how well the neural network can detect the target from far away
```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.995441234028656
average intersection over union for other people is 0.40741392661423764
average intersection over union for the hero is 0.19374283779449622
number true positives: 118, number false positives: 0, number false negatives: 183
```
### Sum all the true positives, etc from the three datasets to get a weight for the score
```
0.7365470852017937
```
### The IoU for the dataset that never includes the hero is excluded from grading
```
0.553171242349
```

### And the final grade score is 
```
0.40743666617
```

# Test the model that have been created in the quadcopter simulator

The model weights selected is model_weights_new that have final score 40.74, to run this model weight

```bash
>python follower.py model_weights_new
```
[[Simulation Video]](https://youtu.be/cotA3RwEjA8)

# Future Enhancement
For future enhancement, there are several thing that need to be improved to increased final model score and accuracy in simulator that is:
1. Increased Data Training.
In this project I didn't record any data train manually except I used data train that provide by udacity. To get more data train, I must add data train that provided by udacity with data train that I will collecting manually.
2. Decreased Time Training
I have training the model using my laptop which have a standard graphical card which give me about 22.8 hour to finish model training. Because my EC2 Instances limit increase request have been approved by AWS, I will train my model using AWS Services.
3. Change Hyperparameters
In this project, I used epochs and steps_per_epoch limited to get passing required scores. I need to increase the number of epochs to increase my model final scores.

References:
* https://www.researchgate.net/publication/277335071_A_Bottom-Up_Approach_for_Pancreas_Segmentation_Using_Cascaded_Superpixels_and_Deep_Image_Patch_Labeling
* https://www.researchgate.net/publication/220785200_Efficiency_Optimization_of_Trainable_Feature_Extractors_for_a_Consumer_Platform
* https://iamaaditya.github.io/2016/03/one-by-one-convolution/
* Network in Networks Paper : arxiv.org/pdf/1312.4400v3.pdf
* https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html