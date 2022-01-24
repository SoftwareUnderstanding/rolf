# Ear Disease Detection

## Introduction

Otitis Media (OM) is an infection of the middle ear. It is one of the most common childhood illnesses and the second most important reason leading to the loss of hearing. It is most common in developing countries and was ranked fifth on the global burden of disease and affected 1.23 billion people in 2013.

OM is often misdiagnosed or not diagnosed at all, especially when it is in the early stages. It is often either under-diagnosed or over-diagnosed depending on the factors like clinicians, symptoms, otoscopes etc. Detection of OM requires a good medical practitioner (ENT), whose availability is difficult in remote village areas especially in developing countries. That is why OM is ignored amongst these kinds of groups and is a second major cause of hearing loss.

The aim of the study is to develop a diagnostic system using Ear Drum (Tympanic Membrane) images and applying machine learning to automatically extract certain features and perform image classification which can help diagnose otitis media(OM) with greater accuracy.
This diagnostic system will provide a reliable data to a survey volunteer to advise the patient or his family to visit an ENT or take professional help if OM is present.


<img src="https://3.bp.blogspot.com/-WBPelBryAoE/WrJhXwz5XtI/AAAAAAAAEPA/iMSU4TXcNWIe7jK2G3P6xo4Ls4DWisbTgCLcBGAs/s1600/wix%2B11.jpg" width="400" title="Anatomy of the Ear">


## Dataset Description

One of the biggest challenge faced during the project was the collection of image data for normal and infected tympanic membrane. We obtained the ear disease dataset from an ENT doctor at AIMS. The dataset consists oof around 250 images with 6 classes. Details of the classes: 

![alt text](images/dataset.JPG?raw=true)

### Glimplse into the dataset    

<table>
    <tr align = "center">
      <td><img src="https://github.com/SarthakGarg13/OTOMYCOSIS/blob/master/images/normalear.JPG" width="200"/></td>
<td><img src="https://github.com/SarthakGarg13/OTOMYCOSIS/blob/master/images/glue_ear.jpg" width="205"/></td>
<td><img src="https://github.com/SarthakGarg13/OTOMYCOSIS/blob/master/images/otomycosis.jpg" width="200"/></td>
    </tr>
    <tr><td align="center">Normal Ear</td><td align="center">Glue Ear</td><td align="center">Otomycosis</td></tr>
</table>


## Building a CNN

Now we are ready to build a CNN. After dabbling a bit with tensorflow, I decided it was way too much work for something incredibly simple. I decided to use keras. Keras is a high-level API wrapper around tensorflow. It made coding lot more palatable. The approach I used was similar to this. I used a 3 convolutional layers in my architecture initially.

![alt text](images/cnn.png?raw=true "CNN Architecture")

## Transfer Learning
Transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem. For instance, features from a model that has learned to identify racoons may be useful to kick-start a model meant to identify tanukis.

Transfer learning is usually done for tasks where your dataset has too little data to train a full-scale model from scratch.

The most common incarnation of transfer learning in the context of deep learning is the following worfklow:

- Take layers from a previously trained model.
- Freeze them, so as to avoid destroying any of the information they contain during future training rounds.
- Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.
- Train the new layers on your dataset.


```
tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs)
```
Instantiates the ResNet50 architecture.

Using keras.applications for initilizing model architecture with imagenet weights. Similarly we used InceptionResnetV2 and VGG16 from keras applications and used imagenet weights for initialization.

## Results

<p align = "center">
<table>
    <tr>
      <td align="center"><img src="https://github.com/SarthakGarg13/OTOMYCOSIS/blob/master/images/retracted.JPG" width="200"/></c></td>
<td align="center"><img src="https://github.com/SarthakGarg13/OTOMYCOSIS/blob/master/images/wax.jpg" width="200"/></td>
    </tr>
    <tr><td align="center">Retracted Typanic Membrane</td><td align="center">Wax</td></tr>
</table>
</p>

## Deployment

We have developed a webapp using Flask API and have deployed it on Heroku that enables us to operate entirely on cloud.
A front-end for the website has been deployed on Netlify which could be accessed by [clicking here!](https://otology.netlify.com)

![alt text](images/website.png?raw=true "Prototype Website")

## References

- https://core.ac.uk/download/pdf/161426117.pdf
- https://www.sciencedirect.com/science/article/pii/S2352396419304311
- https://arxiv.org/abs/1512.03385
- https://medium.com/bhavaniravi/build-your-1st-python-web-app-with-flask-b039d11f101c
- https://keras.io/api/applications/inceptionv3/




