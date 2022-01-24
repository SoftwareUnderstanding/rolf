# Image's Object Detection
## Abstract
The application uses TensorFlow Object Detection API and Flask Python to create an application for image detection. User input image and the object detection algorithm will return the objects in the image. All the images and objects will be saved into the database. The application also detect objects from stream video, and capture the stream video with its object.

## Video Overview
https://www.loom.com/share/5ade6e77f982440a9356c3e12457c612

## Functionals Requirements
- The application will implement user registration module, login module, and object detection module. <br/>
- The application separate presentation layer, business layer and database layer. <br/>
- The application separates the view (jinja2 templates) and the controllers. <br/>
- The application uses TensorFlow Object Detection API to detect image’s objects. <br/>
- The application use relational database MySQL. <br/>
- The application is deployed on Flask server and localhost. 


## Logical Solution Design
The solution divides the application to Controller (Python site) and View (Jinja2 templates). The server side includes a Flask server and TensorFlow Object Detection API.

![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/logical.png)
## Physical Architecture
The Flask server, TensorFlow Object Detection API, and MySQL database are hosted and demo on localhost (Windows OS CPU 2.2GHZ x 8GB GPU x 16 GB RAM x 16 GB Storage). 
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/physical.png)

## Component Design
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/component.png)

## Deployment Diagram
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/deployment.png)

## Database
A high-level view of the database, showing how the tables relate to each other and what rows there are in each table.
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/ER.png)

## Sitemap
An overview of the flow of the website.
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/sitemap.png)

## Technical Design
A brief overview of the general approach and the main libraries that were leveraged can be found [here](https://github.com/chuongngd/Images-Object-Detection/blob/master/docs/Technical%20Design.md)
## UML Class Diagram
A brief overview of class and functions of the application 
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/uml.jpg)
## Modules
1. [Login and Registration](https://github.com/chuongngd/Images-Object-Detection/blob/master/docs/Login%20and%20Registration.md)
2. [Image Detect Module](https://github.com/chuongngd/Images-Object-Detection/blob/master/docs/ImageDetection.md)
3. [Video Detect Module](https://github.com/chuongngd/Images-Object-Detection/blob/master/docs/videodetection.md)

## Installing
Because the application haven't been deployed on cloud. To installing and testing the application on localhost, user needs to install TensorFlow environment, MySQL, Flask and Python 3.6

## Apply on live system

This application can be applied in security monitoring, traffic tracking, or counting products in manufactury. <br/>
For example, an officer is looking for a stolen car on the street. If there are system cameras on the street, the officer
can search the suspect easier and tracking it. 

## Conclusions
Image processing is very important in every aspect of life. Detection algorithm, dataset for images, support libraries, etc. are still reaserching and developing. In this project scope, to apply it in real life, there are still limits that needs to develope. 
## Future Ideas
Deploy on cloud. <br/>
Running Object Detection on cloud machine learning that can increase the performance. <br/>
Increase the dataset that can detect more object's category. <br>
## Built With

* [Flask](http://flask.pocoo.org/) - The web framework used
* [Ananconda](https://www.anaconda.com/distribution/) - The distribution to support TensorFlow and Python packages
* [TensorFlow](https://www.tensorflow.org/) - The open-source software library supports machine learning
* [MySQL](https://www.mysql.com/) - The open-source relational database management system supports for store database in the project
* [Python](https://www.python.org/) - The programming language use for the project

## References

1.	Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. You Only Look One: Unified, Real-Time Object Detection. University of Washington, Allen Institute of AI, Facebook AI Research. Retrieved November 2, 2018, from https://arxiv.org/pdf/1506.02640.pdf <br/>
2.	Hui, J. (2018, Mar 17). Real-time Object Detection with YOLO, YOLOv2, and now YOLOv3. Retrieved from https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088 <br/>
3.	Tran, D. (2017, Jul 28). How to train your own Object Detector with TensorFlow’s Object Detection API. Retrieved from https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 <br/>
4.	Hard, C. (2017, Dec 3). Computer Vision on the Web with WebRTC and TensorFlow. Retrieved from https://webrtchacks.com/webrtc-cv-tensorflow/ <br/>
5.	Deploy a Flask Application on an Azure Virtual Machine. Retrieved from http://leifengblog.net/blog/deploy-flask-applications-on-azure-vps/ <br/>
6.	TensorFlow project. https://github.com/tensorflow <br/>
7.	The PASCAL Visual Object Classes Homepage. Retrieved from http://host.robots.ox.ac.uk/pascal/VOC/ <br/>
8.	https://www.tensorflow.org/ <br/>
9.	https://www.tensorflow.org/tutorials/ <br/>
10.	https://opensource.google.com/projects/tensorflow <br/>
11. [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)



## Authors

* **Chuong Nguyen** 


