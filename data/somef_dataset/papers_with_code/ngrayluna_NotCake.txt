
<img class="center" src="./imgs/small.png">

<!-- <h1>NotCake</h1> -->

<h2><a name="about">About</a></h2>  
NotCake is a Flask web application that uses Artificial Intelligence to detect whether an image contains instances of cake. The application works by using a pre-trained Convolutional Neural Network for object recognition. 

This is project is currently under development. Currently, the application can take an uploaded photo and make predictions. A future release will allow the user to take a photo with the NotCake App directly.  

This repository holds both the underlying Deep Neural Network along with the Flask Python and HTML files used to create the application.    


The pipeline which will be included and fully documented in this repository will consist of:  


<b>How to Downloading Image Data</b> --> <b>Pre-Processing & Formatting</b> --> <b>Defining a Model</b> --> <b>Compiling a the Deep Neural Network</b> --> <b>Training</b>  


<h2>Table of Contents</h2>  

* [Dependencies](#dependencies)  
* [How to Use](#how)  
* [Acknowledgments](*acknowledgments)
* [Links](#links)   



<h2><a name="dependencies">Dependencies:</a></h2>  

  
<!-- All scripts are written in Python3. The following libraries (and their version) were also used:   -->

* Numpy                1.18.1   
* Flask                1.1.1 
* opencv-python        4.1.2.30
* Keras                2.1.4  
* TensorFlow 1.8.0  
* Matplotlib  

<h2><a name="how">How to Use</a></h2>  

First, make sure you have all the right dependencies. 

Second, upload your images into the “imageDetect” folder (this is only temporary). 

You’re all set. Next run:

```python
python app.py
```

<img src="./imgs/HomeImage.png">

<h2><a name="acknowledgments">Acknowledgments</a></h2> 
* [Huynh Ngoc Anh's](https://github.com/experiencor) implementation of the YOLO framework, and its associated dependencies.  


<h2><a name="links">Links</a></h2>  
* [Paper on YOLO](https://arxiv.org/pdf/1506.02640.pdf)


