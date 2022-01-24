**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Introduction](#introduction)
  - [What problem does ResNet solve](#what-problem-does-resnet-solve)
  - [How does he solve it](#how-does-he-solve-it)
  - [Architecture of ResNet](#architecture-of-resnet)
 - [What is Pneumonia](#what-is-pneumonia)
 - [Dataset](#dataset)
   - [Data exploration](#data-exploration)
   - [Data resizing](#data-resizing)
   - [Data augmentation](#data-augmentation)
 - [Transfer learning](#transfer-learning)
   - [Why transfer learning](#why-transfer-learning)
 - [Results comparison](#results-comparison) 
   
# Introduction 
<br/>

According to the universal approximation theorem, with sufficient capacity, we know that a feedforward network with a single layer is sufficient to represent any function but the layer can be massive and the network tends to overfill the data. Therefore, there is a common trend in the research community that our network architecture needs to go further. However, increasing the depth of the network doesn't work by just stacking the layers together. Deep networks are difficult to form due to the notorious problem of the leakage gradient - as the gradient is back propagated to the earlier layers, repeated multiplication can make the gradient infinitely small.
<br/>

## What problem does ResNet solve
<br/>

One of the problems solved by ResNets is gradient leakage. Indeed, when the network is too deep, the gradients from which the loss function is calculated easily reduce to zero after several applications of the chain rule. This result on the weights never updates its values and therefore no learning is performed.

<p align="center">
    <img src="screenshots/resnetcourbe.png" width="700" height="300">
</p>

<br/>

## How does he solve it
<br/>

Instead of learning a transformation of  x -> y  with a function  H(x)  (Some stacked nonlinear layers). Define the residual function using  F(x) = H(x) - x, which can be cropped to  H(x) = F(x) + x, where F(x) and x represent stacked nonlinear layers and identity function (input = output) respectively.

The author's hypothesis is that it is easy to optimize the residual function F (x) rather than to optimize H (x).

The central idea of ResNet is to introduce a so-called "identity shortcut connection" which skips one or more layers, as shown in the following figure:
<p align="center">
    <img src="screenshots/resnet_bloc.png" width="700" height="300">
</p>
<br/>

## Architecture of ResNet
<br/>

Since ResNets can have varying sizes, depending on the size of each layer in the model and the number of layers it has, we will follow the authors' description in this article : https://arxiv.org/pdf/1512.03385.pdf to explain the structure after these networks.

<p align="center">
    <img src="screenshots/resnet_structure.png" width="700" height="1100">
</p>

Here we can see that the ResNet (the one below) consists of a convolution and grouping step (on orange) followed by 4 layers of similar behavior.

Each of the layers follows the same pattern. They perform a 3x3 convolution with a fixed feature map dimension (F) [64, 128, 256, 512] respectively, bypassing the input every 2 convolutions. In addition, the width (W) and height (H) dimensions remain constant throughout the layer.

The dotted line is there, precisely because there has been a change in the size of the input volume (of course a reduction due to convolution). Note that this reduction between layers is obtained by increasing the stride, from 1 to 2, at the first convolution of each layer; instead of through a pooling operation, which we're used to seeing as down samplers.

In this table, there is a summary of the output size at each layer and the dimension of the convolutional karnel at each point of the structure :

<p align="center">
    <img src="screenshots/resnet.png" width="700" height="300">
</p>

In what follows we will try to compare the performance of a ResNet vs a other normal CNNs on a Dataset.
<br/>


# What is Pneumonia 
<br/>

Pneumonia is an inflammatory condition of the lung mainly affecting the small air sacs called alveoli. Symptoms usually include a combination of productive or dry cough, chest pain, fever, and difficulty breathing. The severity of the condition varies.

Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain drugs, or conditions such as autoimmune disease. Risk factors include cystic fibrosis, lung disease Chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, poor ability to cough as a result of a stroke and a weakened immune system. Diagnosis is often based on symptoms and physical examination.

Chest x-ray, blood tests, and sputum culture can help confirm the diagnosis. The goal of this project is to train a ResNet model to help us detect Pneumonia from chest x-ray.


<p align="center">
    <img src="screenshots/pneumonia.jpg" width="700" height="300">
</p>

<br/>

# Dataset 
<pre>
Dataset Name     : Chest X-Ray Images (Pneumonia)
Dataset Link     : <a href=https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Chest X-Ray Images (Pneumonia) Dataset (Kaggle)</a>
                 : <a href=https://data.mendeley.com/datasets/rscbjbr9sj/2>Chest X-Ray Images (Pneumonia) Dataset (Original Dataset)</a>
Original Paper   : <a href=https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5>Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning</a>
                   (Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, M. Anthony Lewis, Huimin Xia, Kang Zhang)
                   https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
</pre>

<br/>

The Dataset is organized in 3 folders (train, test, val) and contains sub-folders for each image category (Pneumonia / Normal). There are 5863 X-ray images (JPEG) and 2 categories (Pneumonia / Normal). Chest (anteroposterior) x-rays were selected from retrospective cohorts of pediatric patients aged one to five years at the Guangzhou Women's and Children's Medical Center in Guangzhou. All chest x-rays were taken as part of routine clinical patient care. For analysis of chest x-ray images, all chest x-rays were initially reviewed for quality control by removing any poor quality or unreadable scans. The diagnoses for the images were then scored by two expert doctors before being validated for training the AI system. In order to take account of possible scoring errors, the evaluation set was also checked by a third expert.
<br/>

## Data exploration
<br/>

We have two classes, Pneumonia and Normal. The data appear to be out of balance. To increase the normal training examples we will use data augmentation.
<p align="center">
    <img src="screenshots/dataset_balance.png" width="500" height="300">
</p>

<br/>
<br/>

Some images from the Dataset : 

<p align="center">
    <img src="screenshots/images.png" width="500" height="300">
</p>

<br/>

## Data resizing
 <br/>

Now back to our X.train and X.test. It is important to know that the shape of these two arrays is (5216, 224, 224) and (624, 224, 224) respectively. Well, at a glance these two shapes look good as we can just display them using the plt.imshow () function. However, this shape is simply not acceptable to the convolutional layer as it expects a color channel to be included as an input.

So, since this image is mostly colored in rgb, then we have to add 3 new axes with 3 dimension which will be recognized by the convolution layer as the color channels.
<br/>

## Data augmentation 
<br/>

In order to avoid any overfitting problem, we need to artificially expand our dataset. We can make your existing dataset even bigger. The idea is to modify the training data with small transformations to reproduce the variations. Approaches that modify training data to change the representation of the table while maintaining the same wording are called data augmentation techniques. 
Some popular augmentations that people use are : grayscale, horizontal inversions, vertical inversions, random crops, color jitter, translations, rotations, and much more. By applying just a few of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

For the increase of data we have chosen:

- Random zoom of 0.2 on some training images
- Randomly rotate some training images by 30 degrees
- Shift images horizontally randomly 0.1 of the width
- Shift images vertically randomly 0.1 height
- Randomly flip the images horizontally.

<br/>

# Transfer learning
Transfer learning is a machine learning problem that focuses on retaining the knowledge gained by solving a problem and applying it to a different but related problem. For example, the knowledge gained from learning to recognize cars could be applied when trying to recognize trucks.

Indeed, one of the great motivations of transfer learning is that it takes a large amount of data to have robust models (especially in deep learning). So, if we can transfer some knowledge acquired during the creation of an X model, we can use less data for the creation of a Y model.

## Why transfer learning
Because with transfer learning, you start with an existing (trained) neural network used for image recognition - then modify it a bit (or more) here and there to train a model for your case. particular use. And why are we doing this? Training a reasonable neural network would mean needing around 300,000 image samples, and for very good performance we would need at least a million images.

In our case, we have around 4000 images in our training set - you have a guess as to whether that would have been enough if we had trained a neural network from scratch.

We are going to load a pre-trained networks, which have been trained on approximately one million images from the ImageNet database. The models that we are going to use are : 
 -  VGG16 : The VGG-16 is one of the most popular pre-trained models for image classification. It was and remains THE model to beat even today. Developed at the Visual Graphics Group at the University of Oxford, VGG-16 beat the then standard of AlexNet and was quickly adopted by researchers and the industry for their image Classification Tasks.

 -  Inception V3 : the Inception model module just performs convolutions with different filter sizes on the input, performs Max Pooling, and concatenates the result for the next Inception module. The introduction of the 1 * 1 convolution operation reduces the parameters drastically. Though the number of layers in Inceptionv3 is 24, the massive reduction in the parameters makes it a formidable model to beat.
 
 -  ResNet50 : And last, we are going to test the ResNet interduced earlier and see how it compares to the other models. 


# Results comparison


| Model name         |    Precsion     |     Recall     |    F1 score    |
| ------------------ |---------------- | -------------- | -------------- |
| VGG16              |     91%         |      90%       |       90%      |
| Inception V3       |     88%         |      88%       |       87%      |
| ResNet 50          |     91%         |      90%       |       90%      |

- the VGG16 model : Our were able to acheive accuracy with only 10 epochs that is good results. However, our modelstarts overfitting after that and we can't improve our accuracy any longer.
- The Inception model : The accuracy was less than for VGG16. However, the training took less time. But the model started overfitting aswell after.
- What we noticed, is that these two models overfit quite rapidly which makes it impossible to learn more unless we have more data. 
- For ResNet model however, the problem of over fitting is resolved, and we could acheive 92% accuracy rate. 

