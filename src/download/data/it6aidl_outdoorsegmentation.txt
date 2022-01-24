
## Repository Structure

The repository is structured as follows:
- experiments -  folder containing an example of the experiments conducted
 - figures - folder containing project images.
 - toolbox - folder containing two auxiliar tools for working with the dataset: transforming the labels merging some of the classes, creating csv files for loading the images in the dataset and getting stats for class imbalance.
 - Dataset.py - python file containing the dataset class.
 - Train.py - python file containing the train class.
 - Test.py - python file used for doing only the Test step.
 - Unet.py - python file containing the Unet class.
 - UnetModes.py - python file containing extra Unet modes.
 - metrics.py - python file containing metric calculators.

# Outdoor Semantic Segmentation

### About
 - Date: 14/4/2020
 - Authors: Alberto Masa, Fernando Tébar, Mariela Fierro, Sina Sadeghi
 - Institute: Universitat Politecnica De Cataluña

## Motivation

Based on our team's collective interest in computer vision we decided to pursue a semantic segmentation task, using deep learning to classify objects in road scenarios to further the development of self-driving cars.

## Proposal
 - [x] Analyze the data provided in the selected dataset and adapt it to be used in a Semantic Segmentation network.
 - [x] Mitigate the class imbalance based on a better understanding of our data.
 - [x] Learn how to do a transfer learning from the previous task to another one, for instance, detecting the drivable area.
 - [x] Reproduce a semantic segmentation network described in the U-Net paper from scratch.
 - [x] Apply data augmentation, generate different kinds of weather such as fog, rain, snowflakes.

## Milestones

 - [x] Obtain and process the Cityscapes dataset.
 - [x] Train a semantic segmentation network and analyze the results.
 - [x] Use weighted Loss.
 - [x] Apply data augmentation.
 - [ ] Use model for selecting drivable area.

## Dataset
The Cityscapes dataset includes a diverse set of street scene image captures from 50 different cities around the world designed specifically for training segmentation models. The dataset includes semantic, instance-wise, and dense pixel annotation for a total of 30 classes. The dataset consists of 5,000 images at a resolution of 1024x2048.

![Cityscapes](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Cityscapes.png)

A custom dataset class capable of loading the images and targets from any dataset was created. Thus, in order to increase the flexibility of the trainings, a [Split_Url_Generator](https://github.com/it6aidl/outdoorsegmentation/blob/master/aux/Split_Url_Generator.py) was created to produce a .csv files containing the URLS linking the dataset images and targets for Train, Val and test. This way, this Dataset class can be reused for any other dataset, just generating new .csv files for the new dataset.

The following functions where included,

-   init( ) - initializes dataset object.
-   len( ) - returns dataset size.
-   getitem( ) - loads and transforms image/target.

This Dataset class is also prepared to apply different kind of transformations for data augmentation. These transformations can be applied only to the images or to both the image and the target, controlling that the random transformations are applied at the same time. In addition, this class includes the weather transformation used for ensure that the model is prepared to work in any climatological conditions using [imgaug](https://imgaug.readthedocs.io/en/latest/).  

By default, the loaded images are resized to 256x512 and converted to tensors during the transformation. The loaded targets are also resized to 256x512, with the interpolation parameter set to 0. This resizing has been done for reducing the training resources required.

A snippet of the transformation code and data loaders is presented below,

~~~
  train_dataset = MyDataset(version=ds_version, split='train', joint_transform=joint_transforms, img_transform=img_transforms, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'])
  train_loader = utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)

  val_dataset = MyDataset(version=ds_version, split='val', joint_transform=joint_transforms_vt, img_transform=img_transforms_vt, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'], add_weather= weather == 'y')
  val_loader = utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)

  test_dataset = MyDataset(version=ds_version, split='test', joint_transform=joint_transforms_vt, img_transform=img_transforms_vt, url_csv_file=params['dataset_url'], file_suffix=params['file_suffix'], add_weather= weather == 'y')
  test_loader = utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
~~~

In particular, we worked with CityScapes Fine annotations and 8-bit images. The following number of images and targets where used for each split:
-   Test: 250
-   Validation: 250
-   Train: 2952

As suggested by the Cityscapes documentation, classes with a label id of 255 were emitted. Also, we used another [tool](https://github.com/it6aidl/outdoorsegmentation/tree/master/toolbox) to merge some of the classes with less representation, resulting in a total of 19 distinct classes: Road, Sidewalk, Building, Wall, Fence, Pole, Traffic Light, Traffic Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, and Bicycle. In order to know the kind of data we are working with, we created aux functions to calculate the [statistics](https://github.com/it6aidl/outdoorsegmentation/blob/master/aux/dataloader_stats.py) of the train and validation splits. This information was helpful to understand the class imbalance so that we can apply techniques to deal with it that we will explain above.

![Class stats](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Class_Stats.png)

## Architectures
### Network
#### U-net
The U-net is a fully convolutional network created specifically for computer vision problems, including segmentation tasks. It became popular because of its efficient use of the GPU and its ability to learn with a limited dataset. What makes the U-net noble from other network architectures is that every pooling layer is mirrored by an up-sampling layer. The following figure shows the U-net contracting path (left side) and an expansive path (right side), both of which are symmetrically structured.

![Unet Model Diagram](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet.png)

This allows the network to reduce spatial information while increasing feature information on its way down, and reduce feature information while increasing station information on the way up, leading to highly efficient image processing.

##### Concatenation Layer
Since the U-net downsamples the feature information in the first half of the network, there is a risk of loosing valuable information. To overcome this, we concatenated all the feature maps in the decoding layers with the feature maps from the encoding layers. This assures that any information learned in the ignitions layers will be retained throughout the network.

![Concat Layers](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Unet%2BConcat.png)

##### Bi-linear Interpolation
In order to recover the original input resolution at the output of the network, a bi-linear interpolation was performed. For bi-linear interpolation, a weighted average is calculated on the four nearest pixels to achieve a smooth output. The data was interpolated along 2-axis during upsampling, following the following formula,

![Bilinear Formula](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Bilinear%20Interpolation.png)

where f(x,y) is the unknown function, in our case pixel intensity, and the four nearest points being,

![Bilinear Formula2](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Bilinear%20Interpolation2.png)

##### Transposed Convolutions
To improve the quality and efficiency of the upsampling, the bi-linear interpolation was replaced by transposed convolutions. These convolutions make use of learneable parameters to enable the network to “learn” how to best transform each feature map on its own.

#### DeepLabv3
As a way to test a different approach, and trying to improve the results, we tested a different model: DeepLabv3. Here, we picked the version already existing and pretrained in Torchvision. This architecture review how atrous convolutions are applied to extract dense features for semantic segmentation and how to use a combination of them in parallel under the concept of Atrous Spatial Pyramid Pooling.  

![Deeplabv3](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/deeplabv3.png)


### Optimizer
An optimizer is necessary for minimizing the loss function. In this project both the Adaptive Moment Estimation (Adam) and Stochastic Gradient Descent (SGD) optimizers were tested. SGD calculates the gradient descent for each example, reducing batch redundancies, and improving computational efficiency. Adam is a mixture of the SGD and RMSprop optimizers, and offers an adaptive learning rate, increasing the network's flexibility.


### Metrics

Evaluating and comparing the experiments is a nuclear part of the scientific work and opens the path to adjust parameters and propose changes. For this project we defined several metrics to compare models and trainings

#### Pixel accuracy

The main metric we used to evaluate the experiments if the accuracy of the model configuration. The model prediction accuracy is calculated dividing the number of encerted pixels by the number of total pixels. However, there is a class that we are ignoring throughout the experiments and does not compute for the accuracy. This class represents objects in the images that are not useful for our purposes (thrash cans and other street objects)

#### IoU per class
The previous metric is a generalization of how our model works overall. This next one gives emphasis on the nature of the data. Intersection over Union (Jaccard score) is an effective, well known metric for pixel classification in object segmentation problems. The IoU score for each class is a number between 0 and 1 calculated by dividing the intersection from the pixels in prediction and GT by the union of these two surfaces.

#### mIoU

The other metric that illuminated our grievous path through the fathomless darkness of semantic segmentation was the mean Intersection over Union. A mean calculation of every class IoU is used to measure how well is the model classificating all the classes.

## Results

Throughout the process of building a strong model, multiple experiments were conducted in order to track progress. We stuck with the configuration that gave us better results until the moment and build on top of it.

### Experiment 1: UNet (without concats)

In the very beginning, we decided to build an easy-to-code, lightweight model that worked for us to see a nice segmentation result easy to understand and easy to be improved by adding components to the configuration. It also gave some hints about the base results we could achieve.
It was intended as well to act as a base-line for the all future experiments. It consisted of a linear version (removing the concatenations) of the UNet. Easy as it was, such a model casts very little precision to the prediction. For this implementation, we decoder phase made use of *torch.nn.Upsample* to upsample the encoded features.
As an optimizer we chose Adam because it needs no additional tuning or adjust hyperparameters.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamlinearloss.png)

#### Results

| Optimizer (LR) | Model | Version | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|
| Adam (0.001) |  UNet| no concats|82.25 | 40.1

### Experiment 2: UNet full
For the second experiments, we improved the network to embrace the concatenations defined in the canonical net. After achieving a not so bad accuracy result of 75%, we moved on to reproduce the full UNet. This of course turned into computational and practical adjustments such as reducing the batch size and we had to wait longer for the experiment to give results.
The UNet as it was created in [the paper](https://arxiv.org/abs/1505.04597) will give us more precision in the predictions since it adds every phase result of the encoder to the decoder to produce the output. Increase of prediction is really evident.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adambilinearloss.png)

The results, apart from the quantitative side, have shown a real qualitative effect in the sharpness of the predictions, as can be seen in the following comparison. In the left hand side we can see the predictions from the model with the concatenations whereas in the right hand side we can see the results of the architecture without these conections.

![Effect of concatenations](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/effectofconcatenation.png)

#### Results

| Optimizer (LR) | Model | Version  | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|
| Adam (0.001) | UNet|  Bilinear 1x1| 83.46 | 43.23


#### Bilinear Interpolation:  Kernel 3 Padding 1 vs Kernel 1

As earlier said, we used *torch.nn.UpSample* as the component to upsample the encoded features.
We did modify the original experiment substituting the transposed convolutions for the convolutions Kernel size 1 in the last step. However, we wanted to test if using a wider kernel (3x3), it can keep better the spatial information for the feature extraction than the orginal 1x1 one.

~~~
    #New
    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= 1))
    #original
    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                              nn.Conv2d(in_channels, out_channels, kernel_size=1))
~~~

However, the results were not good enough and seems to not support this hypothesis. 

#### Results

| Optimizer (LR) | Model | Version | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|
| Adam (0.001) | UNet| Bilinear 3x3/1|83.13 |41.7



#### Transposed convolutions
There are several other methods to perform the upsampling and we chose the Transposed convolutions. This generated new feature maps double sized in the decoder phases so we end up having the same output size. In practical, the accuracy rised a bit but it was not visible looking at the predictions. The results are shown in the following graph. As these were our better results, we keeped this configuration as our reference.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransloss.png)

#### Results

| Optimizer (LR) | Model | Version  | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|
| Adam (0.001) | UNet|Transpose| 83.64|44.01  


### Experiment 3: Change optimizer

For our next experiment and before introducing other techniques, we decided to change the optimizer to SGD to see how it performed. Comparing validation results we would keep Adam optimizer.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgdtransloss.png)

#### Results

| Optimizer (LR) | Model | Version | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|
| SGD (0.001) | Unet | Transpose| 80.89|34.26



### Experiment 4: Data Augmentation

After stablishing the UNet version and optimizer that gave us better results, we could start experimenting other techniques to boost the model prediction. Altough this technique it is often used to reduce overfitting, have not suffered such, we wanted to test our net and see how this  affects accuracy in validation and test. The transformations done to the images comprehend random horizontal flips and modifications to brightness, contrast, saturation and hue.
Training the network with this technique makes it generalize better on new samples but it gave no better results. As a matter of fact, the test results, where we was expecting a change, suffered a decrease in both metrics.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransdaloss.png)

#### Results

| Optimizer (LR) | Model | Version | Configuration | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) | UNet | Transpose|DA | 82.77| 41.33



### Experiment 5: Inverted Weights
The following experiment was oriented to tackle the class imbalance mentioned above. Thus, using the number of pixels calculated for each class, eeights were added to the loss. The method is explained below:

Based on the number of pixels for each class calculated:
~~~
Number of pixels per class (train):
[127414939, 21058643, 79041999, 2269832, 3038496, 4244760, 720425, 1911074, 55121339, 4008424, 13948699, 4204816, 465832, 24210293, 925225, 813190, 805591, 341018, 1430722, 43963883]
~~~

the inverted frequency is calculated as the inversion of the normalized number of pixels (normalizing by the total number of pixels). This is used to compensate the imbalance of the classes, as shown in the figure above.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransdainvloss.png)

The results of this experiment were not as good as expected. We also discarded to keep using this technique in our following experiments.

#### Results
These results are obtained using the validation split

| Optimizer (LR) | Model | Version | Configuration | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Transpose|Data augm & Inverted freq |75.14|35.09


### Experiment 6: Weather Augmentation
In order for the network to prepare for varying road scenarios, in this experiment, it was trained while running the weather augmentation online to generate rain and snow.
After running the data augmentation experiment and even though not having valuable results, we decided to include some realistic data augmentation. In our case, driving scenario, would be very helpful to add circumstances that drivers find on the daily. Of course, this should help the model to generalize better in exchange of a decreasing validation accuracy.
The photos were added a layer of one of these elements (rain, snow, clouds, fog) *using python library [imgaug](https://imgaug.readthedocs.io/en/latest/)*.

An example of the effect of this transformations using Fog can be shown here
![weather images](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/Weather_effect.png)

And the results:

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamtransweloss.png)

#### Results

| Optimizer (LR) | Model | Version | Configuration | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Transpose|Weather DA |81.28|38.41


### Experiment 7: Deeplabv3
Next, based on the few improvements achieved, we decided to try a different approach, using an already existing model with pretrained weights. Also, this will give us the experience doing this kind of approach. Thus, as commented in the architecture section, we used DeepLabv3.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/adamdeeplabloss.png)

With the basic configuration we didn't achieve better results. This is the reason we keep trying things in the next experiments.

#### Results

| Optimizer (LR) | Model |  Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|
| Adam (0.001) | Deeplabv3 |82.32|39.47


### Experiment 8: Change Deeplabv3 optimizer
We know that in normal conditions Adam will perform better without fine-tuning the hyperparameters, but we wanted to try SGD to know how it will perform with DeepLabv3.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgddeeplabloss.png)

The results with SGD were better than using Adam, but not achieving the best results we had with U-net.

#### Results

| Optimizer (LR) | Model | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|
| SGD (0.001) | Deeplabv3  | 82.42| 41


### Experiment 9: Change learning rate
At this point we hadn't tune the hyperparameters so we decided to explore how this could affect the configuration. We thought it could lead us to a steeper loss and accuracy curve, but, even though the curves were similar, it performed slightly better than the previous version of deeplabv3 using our classic learning rate 0.001.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgd01deeplabloss.png)

#### Results

| Optimizer (LR) | Model | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|
| SGD (0.1) | Deeplabv3 |83.99|46.64



### Experiment 10: Add weather data augmentation
As the last experiment, we added the same weather data augmentation we performed on our UNet to deeplabv3. The experiment was very disappointing in the validation phase since accuracy dropped to the lowest, but that's the natural answer to data augmentation.

![Loss graph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/lossfigures/sgd01deeplabweatherloss.png)


#### Results
| Optimizer (LR) | Model | Version | Configuration | Pixel accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| SGD (0.1) | Deeplabv3 | |Weather DA | 66.32| 17


The model has been penalized in the validation dataset but will generalize better for new real world samples. If we compare the results of our best model (DeepLabv3 - SGD - lr=1e-1) that achieved a mIoU of 46% in Test, and this model trained with weather conditions, all against the Test split including weather conditions we can see this effect:

|Trained with weather conditions|Pixel accuracy (%)| mIoU (%)| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15| 16 | 17 | 18 |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| Y |61.91 |16|**0.68** |0.26 |0.46 |0.04 |0.01 |0.05 |0.00 | 0.04| **0.44**| 0.06|0.30 |0.15 |0.00 |**0.42** |0.00 |0.0 |0.00 |0.00 |0.02
| N |36.86| 8| **0.40** |0.1 |0.25 |0.01 |0.00 | 0.03| 0.00|0.03 |**0.08** |0.00 |0.22 |0.09 |0.01 |**0.13** |0.00|0.00 |0.00 |0.01 |0.07 |



## Metrics

Evaluating and comparing the experiments is a nuclear part of the scientific work and opens the path to adjust parameters and propose changes. For this project we defined several metrics to compare models and trainings

### Accuracy


Below it is shown the comparison among all the experiments:

![Accuracy validationgraph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/accval.png)

Here, we can see than the best experiment in terms of accur

### IoU per class

As we presented in the dataset statistics, we have a noticeable class imbalance, which ends up in an unbalanced IoU. The classes that appear the most in the dataset (pavement, sky) reach a higher IoU than the ones that appear very few times (signals, traffic lights)


| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 0.80 | 0.64 | 0.84 | 0.25 | 0.12 | 0.39 | 0.32 | 0.49 | **0.87** | 0.44 | 0.86 | 0.43 | 0.21 | 0.83 | 0.11 | 0.22 | **0.00** | 0.07 | 0.50 |


### mIoU

The class imbalance penalises the results, since we have several classes with an IoU of almost 0, but the rest of them achieves an acceptable result.
The highest mIoU is reached by deeplabv3 with a learning rate of 0.1 but at some sudden points it becomes unstable. Then in order are the UNet (bilinear and transpose), linear and the transpose with data augmentation. Some steps lower are the UNet with data augmentation and inverted frequencies and at last the deeplabv3 with weather data augmentation.

![mIoUvalidationgraph](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miouval.png)


### Validation results

| Optimizer (LR) | Model | Version | Configuration | Accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| No concats||82.25 | 40.1
| Adam (0.001) |  UNet| Bilinear 1x1|| 83.46 | 43.23
| Adam (0.001) |  UNet| Bilinear 3x3/1||83.13 |41.7
| Adam (0.001) |  UNet| Transpose| |83.64|44.01  
| SGD (0.001) | UNet| Transpose|| 80.89|34.26
| Adam (0.001) |  UNet| Transpose|DA | 82.77| 41.33
| Adam (0.001) |  UNet | Transpose|DA & IF |75.14|35.09
| Adam (0.001) |  UNet | Transpose|Weather DA |81.28|38.41
| Adam (0.001) | Deeplabv3 | | |82.32|39.47
| SGD (0.001) | Deeplabv3 | | | 82.42| 41
| SGD (0.1) | Deeplabv3 | | |83.99|46.64
| SGD (0.1) | Deeplabv3 | |Weather DA | 66.32| 17



### Test results

The previous metrics were taken in the validation phase of our training. Concluding the experiment we test the model configuration with the test dataset. The results in this phase give us an overall understanding of the performance.

Here we only compare the results of each of our best configurations for each of our models: UNet and DeepLabv3 with and without weather conditions.

| Optimizer (LR) | Model | Version | Configuration | Accuracy (%) | mIoU (%) |
|--|--|--|--|--|--|
| Adam (0.001) |  UNet| Transpose | | 78.28| 43
| Adam (0.001) | Unet | Transpose|Weather DA |75.72 | 35
| SGD (0.1) | Deeplabv3 | | |78.73| 46
| SGD (0.1) | Deeplabv3 | |Weather DA | 61.91| 16




### Other comparisons

In the next figure we can see that after a similar start in the first 15 epochs, and penalized by the low start, SGD has a steeper mIoU curve and might need more epochs to reach Adam's performance in this metric. Also, from the experience of the last experiments, it would have useful to risen the learning rate for SGD optimizer, even though the loss becomes more unstable.

![miouadamvssgd](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miouadamvssgd.png)

---
The next plot show the accuracy from two experiments. In both the model is UNet with the transposed convolutions for upsampling but in the second experiment we add data augmentation for the training dataset.

![acctransvsda](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/acctransvsda.png)

There really is no noticeable improvement after adding this transformations. We expected it also in the test results since they are unseen samples by the model, but we did not get there an improvement either:

| Configuration| Accuracy | mIoU
|--|--| --|
| Normal |78.3  |43
| Data augmentation | 77.3  |39

---

Our of our last experiments was changing the learning rate of the optimizer. We did it on several configurations, both UNet and Deeplabv3 and both Adam and SGD, and here we can notice a real change. The "standard" learning rate is 0.001 and gives a much smoother curve for loss and also metrics. In every experiment used 0.1 as the learning rate all curves became unpredictable.

![miou01vs0001](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/miou01vs0001.png)


## Conclusion

What drove us during the whole process of experimentation was the evidence that came out. What we searched over and over through reading and proposals was improvement. But unfortunately we didn't find much, at least in the amount we expected. We did expect to see self-evident results as we have been seeing throughout the course in the labs and in the ML books, but maybe we had a stroke of reality.
The strategy of building has been incremental. From a small model performing good to bigger models performing better. Our best configuration has succesively changed as we added well known techniques and even though they haven't cast astonishing results we have trusted the evidence. This is the nature of science and engineering.

We have become familiar with the elements that comprise every phase of a DL problem. From choosing the dataset that best fitted our needs to adding cloud layers to images. Coding a model and its environment in such a comprehensive and powerful library as Pytorch has been tough at times but easy overall. It is a sure shot to rely on all the classes and methods that the library offers, and we have only used a tiny part of it.

Tensorboard has become our closest allie. Being able to follow the experiment in real time is essential to contrast executions and take decisions. Scalars, images and best epochs are truly fundamental resources to have the information gathered.

It has been our first dive into a self-driven deep learning project and we have found it challenging. Sure thing we will move on to power up this model with the pending tasks and explore other DL problems.

| Linear | Bilinear|
|--|--|
|![Linear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/linear.gif) | ![Bilinear timeline](https://github.com/it6aidl/outdoorsegmentation/blob/master/figures/bilinear.gif)|


## Future Work

In the initial list of goals we included optimistically other tasks that unfortunately we haven't been able to complete in time. One of them is perform transfer learning to detect in other dataset (Berkeley DeepDrive) the driveable zone with our best configuration.
Focal loss was also amongst them but we invested a lot of time running all the main experiments and did not manage to add it. At last, the pruning of our model has come out recurrently in meetings as a good solution to perform in production environments but it is not an easy technique to implement and with the little time we had left we had to put it apart.

## References
[1]: Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation". CVPR, 2015. https://arxiv.org/abs/1505.04597

[2]: imgaug library https://imgaug.readthedocs.io/en/latest/

[3]: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation". CVPR, 2017. https://arxiv.org/abs/1706.05587
