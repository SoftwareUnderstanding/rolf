# EfficientNet for Gun Detection
# Table of contents
- [Instructions](#instructions)
- [Gun detection system](#gun-detection-system)
- [Model](#model)
- [Database](#database)
- [Training](#training)
- [Evaluation](#evaluation)
- [Conclusion & Future Work](#conclusion--future-work)

# Instructions

1. Download project file(s)
    * [B0 with AveragePool5](/b0_avgpool_5.py) - training script
    * [B0 with Global Pool](/B0_global.py) - training script
    * [B0 with AveragePool9](/B0_AvgPool9.py) - training script
    * [B1 without DropOut](/B1_without_DropOut.py) - training script
    * [Test Script 1](/BB.py) - tests  bounding box + prediction % (1 image)
    * [Test Script 2](/Test.py) - tests only prediction % (of all images in 'data/guntest' folder)
    * [Checkpoint file](/b0_global.pt) - for B0 with Global Pooling
2. Download dataset
    * our dataset can be downloaded from the following link: [data](https://drive.google.com/drive/folders/1GlK7o_nciEY4J3VlHqWvAdK28HBHrDdN?usp=sharing)
3. Put project files and dataset folder in the same directory
    * please ensure you extract all .zip files
4. Navigate to project's directory in console:
    * directory should include: project scripts, 'data' folder, and any .pt files you may choose to download
5. Enter the italicised commands into the console:
    * *python Test.py*  (for testing inference)
    * *python BB.py* (for testing bounding box detection)
    * *python B0_global.py* (training requires CUDA enabled GPU with VRAM of at least 4GB - tested on machine with 8GB VRAM)
    
Notes for training: 
* To end training early, repeatedly press *ctrl+c* in console
* To restart training from presaved checkpoint, edit b0_global.py(line 231): 'restart=0' -> 'restart=1'
* By default, newly trained models will replace pretrained model file b0_global.pt 
* Modify torch.save(checkpoint, 'b0_global.pt') to save to a different file name.
If you do, you will have to modify Test.py (line 200): model = torch.load('b0_global.pt').to(device) - replace b0_global.pt with your own model name
    
    
To add your own test gun images, place them into *data/guntest/guntrain* or *data/guntest/gunval*, as required

***NOTE:* These instructions are for use on a Windows machine. These may not work for Linux, MacOS, or other Operating Systems**


# Gun detection system

Mass shootings are an unfortunate reality in today's world. Stopping mass shootings have proven to be extremely difficult without drastic and extreme measures. We aim to develop a deep-learning-based solution that will help reduce casualties from shootings through early detection and reporting. 

The purpose of our planned system will be to detect guns in videos/surveillance footage and raise an alarm or notify authorities and affected persons if the need arises. Although outside of the scope of this project, our system should be accurate and precise enough to allow for active protection systems to act on our data.

Our planned system will detect guns in a given image/frame and attempt to create a bounding box around any detected guns. However, it should be noted that for the purposes of this project we will focus more on detection than on bounding boxes, so the bounding boxes drawn by our model may not be as accurate as our model's detection accuracy.

# Model
## Overview
We plan to use the EfficientNet architecture to detect guns in real-time. EfficientNet is an architecture that takes advantage of compound scaling (i.e. scaling in Depth, Width and Resolution dimensions) to achieve state-of-the-art accuracy with lower FLOPS and less parameters than models that scale a single dimension. A key point in the original development of this architecture was that the efficiency and accuracy of the model when scaled depends on how much you scale each dimension w.r.t each other. Therefore, the scaling factors (α, β and γ) can be found for the best results when using EfficientNet. Our scaling factors were taken from those used by other implementations. Another key feature of this model that was particularly attractive to us was the low inference times - an important factor to consider when making a gun detector.

Below is a representation of our basal model (EfficientNet B0) without any scaling from the original documentation:

![](graphics/EfficientNetArch.png)

And here is a graphical representation of our model:

![](graphics/EfficientNet_illustration.png)

## Key features
When developing our model, we had to implement the following key features
* Data transforms
* Data loaders
* Swish activation
* MBConv layer
  - Inverted Residual Block
  - Squeeze and Excitation
  - Dropout layer

### Data transforms/augmentations and Data loaders
These features of our model are self-explanatory. They define the transformations we will do to our data, and the directory from which our data is to be downloaded. The code then downloads the images and labels of each class and splits them into training and validation data.

The data augmentations help expose our model to a greater variety of guns (rotated/scaled/perspective), benefiting the generalisation ability of our model. Here is an example of our training augmentations. 
![](graphics/dis.PNG)

### Swish activation
Swish activation is defined as **σ(x) × x**. That is, the sigmoid of x, multiplied by x. The Swish activation function, while it looks similar to the ReLU function, is smooth (i.e. it does not abruptly change its value like ReLU does at x = 0). A graph of this function can be seen below:

![](graphics/Swish.png)

The reason behind using Swish activation for EfficientNet is because it was said to provide better results than ReLU from both the *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks paper* by *Mingxing Tan and Quoc V. Le* as well as a Google Brain study from 2017 by *Ramachandran et al.*. Theoretically, unlike ReLU, Swish does not discard values below 0, hence reducing information loss. Note that the Swish activation was not used solely for MBConv layers, but also for the 
and 3x3 convolutional layers.

These papers can be found at the following links:
* Tan et al.: <https://arxiv.org/pdf/1905.11946.pdf>
* Ramachandran et al.: <https://arxiv.org/pdf/1710.05941v1.pdf>

### MBConv layers
The MBConv layer is the largest component in our EfficientNet implementation. Basically, it is a convolutional layer with Inverted residuals, Squeeze and Excitation, as well as Dropout. These three modules of MBConv are explained below:

#### Inverted Residuals
Inverted residuals firstly have a skip connection, or residual, allowing information and gradients to pass through the block. This allows for deeper models by easing the shrinking of gradients as the model becomes deeper, and also reduces loss of finer information in the earlier layers of the model. The Inverted residual uses another important idea, the 1x1 convolution (pointwise convolution). This convolution expands/decreases the number of feature maps, rather than the resolution. This allows the inverted residual block to increase the feature mapping into a higher dimension at the start of the block. This is so that non-linear activations can be applied within a expanded higher dimension of feature mappings. 

Non-linear activations are essential for neural networks, but can lose information, especially in lower dimensions. The non-linear activation being applied in a higher number of dimensions mitigates the information loss effect. Additionally, depthwise separable convolutions are applied, where convolutions are applied on individual feature map layers. Afterwards, the 1x1 convolution is re-applied to shrink the number of feature mappings, and at the same time achieve spatial feature mapping. The linear output is then added to the skip connection. The use of a depthwise separable convolution increases efficiency by an order of magnitude, by reducing the number of parameters convolved. 

#### Squeeze and Excite
Squeeze and Excite blocks work by *squeezing* an input using Global Average Pooling to the shape  (1, 1, *feature maps*) and multiplying this back to the input. Multiplying the (1, 1, *feature maps*) tensor back to the input increases the weighting of feature maps that have more features, thus '*exciting*' the weightings.

Our implementation of Squeeze and Excite in PyTorch uses the following layers:
* AdaptiveAvgPool2d - this layer computes the (1, 1, *feature maps*) tensor through global average pooling
* Conv2d with ReLU activation - this layer adds non-linearity and reduces output channel complexity
* Conv2d with Sigmoid activation - this layer gives the channel a smooth gating function

The output of the above layers is then multiplied to the input tensor.

#### Dropout layer
The Dropout layer in our implementation works by randomly choosing nodes to deactivate at the end of an MBConv block (does not apply for all the layers, only the ones that are repeated). We chose to add this functionality to our MBConv layers because it helps mitigate the risk of the model 'memorising' data and allows for building new and better connections in the network.

In our testing we attempted two methods of implementing a Dropout layer: we tried using the built-in PyTorch Dropout layer as well as making our own function to randomly choose nodes to drop out. Interestingly, our tests showed that our function outperformed the built-in layer when used in our model. As a result of this testing, we decided upon continuing the use of the function rather than the PyTorch layer for our model.

## Scaling
We used some common scaling factors and other parameters (such as input channels, output channels, and repeats to name a few) to create an Excel sheet that will allow us to calculate the specific parameter to be changed for each model. This allowed us to quickly test different versions of EfficientNet, as the parameters to be changed in each scaled model could be easily found. We were also able to make quick changes to the scaling calculations to test out results from different scaling factors and/or rounding techniques. Examples from our Excel sheet are shown below:

B3 model:

![](graphics/B3_Excel.png)

B1 model (with repeats in raw form, and an extra column with all repeats rounded up):

![](graphics/B1_Excel.png)

The Excel sheet can be found [here](https://drive.google.com/open?id=1trUOhY_HJaVsSx8iwTS-0O_F2NpxE4cJ) (Please download before using)

## Training, Fine-tuning and Optimisation

Over the course of this project we have trained various versions of EfficientNet with different optimisers, learning-rate schedulers, different datasets, and differing numbers of epochs.

The optimisers that we have tested are the Adam optimiser and the Stochastic Gradient Descent (SGD) optimiser. Initially, we were expecting the Adam optimiser to outperform SGD in our tests. However, SGD surprisingly  had better convergence than Adam. As a result, we chose to continue testing SGD. During our tests with SGD, we began experimenting with the use of *Nesterov momentum*.

Nesterov Momentum is a variant of SGD that speeds up training and improves convergence. It basically calculates the gradient term from an intermediate point rather than the current position. This allows for corrections to be made if the momentum overshoots the next point or is pointing in the incorrect direction. See more [here](https://dominikschmidt.xyz/nesterov-momentum/).

In our testing with Nesterov momentum, we found that a momentum of 0.5 works well with our EfficientNet models.

In terms of epoch testing, we have trained our models for varying numbers of epochs ranging from 40 to 224. We found that while our training accuracy percentage easily manages to reach the high 90s within around 40-80 epochs, we needed to train for more epochs in order for our validation accuracy to be similarly as good.

We also found that the use of PyTorch's *ReduceLROnPlateau* learning rate scheduler was useful when our model validation accuracy does not increase for 15 epochs. The reduction in learning rate helps our model finely tune its parameters.

## Extra features
NOTE: Does not work very consistently. Requires input image to be roughly square.

In addition to our EfficientNet model, we decided to create a bounding box algorithm to daw boxes around any guns that we detect. This algorithm works alongside our model but is not actually part of the model itself. Our bounding box algorithm works by way of taking sections from an image containing a detected gun and using a 'sliding window' to check for where in that section a gun may be.

The positions of our sliding window in each input image are as shown below:
![](graphics/BBox.png)

(Note: blue is the area of the sliding window, green is the image. Initially, the program checks the entire image to ensure there is a gun in the input)

After one round of the above sequence, the algorithm checks if the probabilities for guns in any of the boxes are greater than the threshold probability (which is updated to the highest probability found at the end of each sequence). If there is a higher probability, then the section of the image covered by the sliding window for the highest probability will become the input to the algorithm and so on. Eventually, the coordinates of the sliding window with the highest probability will become the coordinates of the bounding box.

Although our bounding box is not as accurate as dedicated single-shot detectors like YOLO or sliding-window approaches which use many sliding windows, our method requires less computation per image – as our goal is real time output, fast inference time is essential. In our testing speed is quite slow but is likely due to model/image loading pipeline constraints, as our model only takes 20-30ms to inference one image, so most of the delay is likely PyTorch loading the model and image. An example of the bounding box algorithm's result can be seen below:

![](graphics/bound2.PNG)

# Database

Our dataset is a custom dataset containing images from Google Images, Gun Wiki, Sai Sasank's dataset, a Synthetic Gun Dataset, COCO and CIFAR10. We have two classes in our dataset, namely *gun* and *not gun*. The *gun* class will contain a combination of gun images from Google, Gun Wiki and the Synthetic Gun Dataset. The *not gun* class will contain a combination of random images from COCO and/or CIFAR10. Each class will have an equal number of images, with the total number of images being *12,250*. This dataset will be further split into training and validation sets. For testing, we have a created a separate set of images (including images from Atulya Kumar and Prasun Roy's datasets) that were not used in our training or testing data. However, our main plan of action is to do real-time testing through a webcam using printouts of gun images.

The reason behind having two classes rather than just a gun class is that we do not want our model to learn that it can get the correct answer by always predicting there is a gun. This would defeat the purpose of the project because the model would predict a large number of *false positives*.

However, we believe that we need an equal number of images in each class (i.e. *balanced* dataset) is to mitigate the chance of our model being biased to either class. This is because, while we do not want our model to guess gun every time, we also do not want it to be biased to the *not gun* class either.

Links to databases used (does not include Gu wiki or Google images):
* [Sai Sasank's dataset](https://www.kaggle.com/issaisasank/guns-object-detection)
* [Synthetic dataset](https://docs.google.com/forms/d/e/1FAIpQLSffVbLwfuhgSvwxrU66NDTZLfz0RrqcQ-KXJxEN9HIZiqxBeg/viewform?vc=0&c=0&w=1)
* [Atulya Kumar's dataset](https://www.kaggle.com/atulyakumar98/gundetection)
* [Prasun Roy's dataset](https://www.kaggle.com/prasunroy/natural-images)
* [COCO](http://cocodataset.org/#download)
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

# Training

Due to hardware constraints and Google COLAB usage limits, our models were only trained for 100 epochs in general. Though, it may be noted that some models were trained for further epochs for the sake of understanding the effect of more epochs on a given model.

### Here is an example of Training Accuracy vs Epochs from our B0 with AveragePool9:

![](graphics/B0_KERNEL9/trnacc_B0.png)

### Here is an example of Training Loss vs Epochs from our B0 with AveragePool9:

![](graphics/B0_KERNEL9/trnloss_B0.png)

We can clearly see from the above graphs that there is a logarithmic relationship between epoch and accuracy/loss.

# Evaluation

Our final four models were limited to using variations of B0 and B1 models as we were limited in terms of time and hardware. The models we have used are outlined below:

Models                                 | Dataset
---------------------------------------|------------------------
B0 with AveragePool95                  | synthetic + real images
B0 with Global Pooling and Adversarial | synthetic + real images
B0 with AveragePool9                   | synthetic + real images
B1 without DropOut                     | synthetic + real images

The results and evaluation for each model are outlined below:

*Note: While the screenshots may say 420 images, there were only 412 images. This was a typo on our end*
### B0 with AveragePool (5x5 kernel):

![](graphics/B0_Avg5/testClassifB0.png) 
![](graphics/B0_Avg5/testConfB0.png)

From the precision results for the B0 with Global Pooling model, we can see that we have a weighted average precision of 0.66. While this is not an inherently bad result, it is unsatisfactory for the purpose of our solution as it means that 0.34 of our positive predictions (for both classes) were false positives. The recall results, similar to the precision results, are good, but not good enough for our purposes. This is because having a weighted average recall of 0.65 means that 0.35 of labelled positives (for both classes) were predicted as false negatives. The F1-score is 0.65 is lower than what we would like to have for our solution. It is also relevant to note that testing resulted in an overall accuracy of 81.67% from 412 images with this network. However, considering that our test set is not perfectly balanced, F1-score may be a better metric to gauge accuracy. This result was not initially expected as this particular model achieved 99.5% validation accuracy in training. However, there was a decline in accuracy when it came to testing because the testing set has images that the model has never encountered before (e.g. cars, motorcycles, etc.), and that have similar features to those the model is trained to detect.

### B0 with Global Pooling and Adversarial training:

![](graphics/B0_Global_Adv/testClassifB0.PNG) 
![](graphics/B0_Global_Adv/testConfB0.png)

From the precision results above we can see a good improvement with the weighted average precision value of 0.89. This means that only a weighted average of 0.11 of positive predictions (for both classes) were false positives. The recall values also seem to be generally better than the previous model, with the weighted average value of 0.87. This means that only 0.13 of labelled positives (for both classes) were predicted as false negatives. The F1-score has also improved to 0.87. It is also relevant to note that testing resulted in an overall accuracy of 87.56% from 412 images with this network. However, considering that our test set is not perfectly balanced, F1-score may be a better metric to gauge accuracy. This model was trained using a special dataset to reduce the adverse effect cars, motorcycles and other test set images have on the other models. This dataset incorporated a slightly more diverse set of images in the 'not gun' class to better narrow down the features that the model should be detecting. As we can see from the results, this model has managed to perform better on the test set. This gives us evidence to support the conclusion that having a more diverse dataset can improve our accuracy.

### B0 with AveragePool (9x9 kernel):

![](graphics/B0_KERNEL9/testClassif_B0.png) 
![](graphics/B0_KERNEL9/testConf_B0.png)

From the above precision metrics, we can see that this model managed a weighted average precision of 0.82 which is the best precision result out of the four models. This means that only 0.18 of the predicted positives (for both classes) were false positives. The recall metric is also better than the other models with a weighted average recall of 0.79. This means that 0.21 of labelled positives (for both classes) were predicted as false negatives. The F1-score accuracy is also the highest at 0.79. It is also relevant to note that testing resulted in an overall accuracy of 84.05% from 412 images with this network. However, considering that our test set is not perfectly balanced, F1-score may be a better metric to gauge accuracy. As originally expected, this model has performed better than the Average Pool 5x5, although the ending validation accuracy for this model was 4% lower than that of the latter (due to limitations in number of epochs trained for).

### B1 without DropOut:

![](graphics/B1_noDrop/testClassifB1.png)
![](graphics/B1_noDrop/testConfB1.png)

From the above precision metrics, we can see that this model has a weighted average precision of 0.62 which is the worst precision result out of the four models. This model has a very unsatisfactory precision, as this means that 0.38 of the predicted positives (for both classes) were false positives. The recall metric is also worse than the other models with a weighted average recall of 0.61. This means that 0.39 of labelled positives (for both classes) were predicted as false negatives. The F1-score accuracy is also the lowest at 0.61. It is also relevant to note that testing resulted in an overall accuracy of 67.62% from 412 images with this network. However, considering that our test set is not perfectly balanced, F1-score may be a better metric to gauge accuracy. The somewhat poor performance of this model was expected as we removed the Dropout layer to see the effect that would have on model performance. From the results of this model we can conclude that the Dropout layer is an important component of the EfficientNet architecture that allows the model to better generalise and fit to the data.

# Conclusion & Future Work
From working on this project, we have learnt/reinforced many concepts and theories, as well as developed a feel for fine-tuning and tweaking features of a model to improve its convergence and overall performance. We have also developed our use and understanding of evaluation tools such as TensorBoard, Confusion Matrices, Precision, Recall, and F1 metrics. All these skills have allowed us to create an EfficientNet model that can accurately detect guns in images/video frames. However, though we have completed this project for the purposes of COMPSYS302, we believe that there is more we can do to improve our model.

Our future aims for this project include the following:
* Use larger models ranging from B3 to B7
* Collate more images to increase the size of our dataset
* Diversify the images in our datasets (including more images of people holding guns, cars, motorcycles, trucks etc.)
* Improving inference times of our models
* Improve the accuracy of our bounding box system
* Improving speed of bounding box generation
* Finding scaling factors that are better suited to our use case
* Run more tests with different versions of pooling to find the optimum combination of pooling parameters


