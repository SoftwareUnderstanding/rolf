# iot_JennyGo


## Columbia University EECS E4764 Fall'19 Internet of Things - Intelligent and Connected Systems

## Team 6 Project Report  JennyGo - Communicate with your turtle!


JennyGo is a product which can make keeping your pet turtle more funny. 
As well as monitoring the phisical state and habitant of turtle, it also predict the mood and personality by machine learning algorithms.
Enjoy talking with your turtle!


### Motivation


Our project aims to enhance the interaction between pet turtle and human. 
With our app, you could “talk” to your turtle and she will response to you based on her probable personality, current activity and mood which predicted by our neural network based on the photos in a real-time method. 
We also implemented environment monitoring to help with building a more comfortable living area for turtle to enjoy. 


### System Architecture
![image](https://github.com/hz2538/iot_JennyGo/blob/master/raspberry/pic01.jpg)

#### Main architecture

Our architecture has four main components: real turtle environment, microcontroller, the back-end cloud and the front-end APP. 
The last three components connect and communicate with each other with http POST and GET. 
The microcontroller loads the temperature and humidity sensor, lighting sensor, and the camera module. 
The cloud loads the Flask platform which provides the interfaces, the AI model as well as our database. 
The APP is the user interface with the main page and the chat page. 
Google speech-to-text (STT) and text-to-speech (TTS) modules are implemented.


#### Technical Components

Monitor mode: 
The temperature, humidity that updated from our sensor every 30 seconds. 
The light sensor controls the lighting condition of the environment, and based on the measurement, our app can switch to the night mode. 
And the camera is running under the “monitor” mode, which capture the turtle every 1 minute. 
It also has a “snapshot” mode, when you click the camera icon, it will provide a snapshot and display on your phone immediately.

Chat mode: 
When you speak to the turtle, it will post a signal to the chip, and switch the camera to a “capture” mode, which would rapidly capture the turtle movements. 
The photos would be sent to the cloud. We hold a flask platform on the cloud, which connect to our AI model and the response database we created. 
We use a deep learning based network called resnet-18 there. We made a transfer learning and adjust the input and output layers. 
For the input, we compress the RGB to grayscale and concatenate 5 frames together as 5 channels of the convolution network to let the model not only learn the spatial frame patterns, but also learn the temperal relationship across the frames. 
For the output layer, the dimension is 7 because we have 7 output status of turtle to be classified: go, stop, turn around, flip over, under the tree, look at the stone, and play the ball.

![image](https://github.com/hz2538/iot_JennyGo/blob/master/raspberry/pic10.jpg)
							
### Deep neural network for status classification:
		
We designed an algorithm to aggregate the movement classification results to judge the mood and personality. 
We have "flat", "excited", "tired", "worry", "think good/bad" for moods, and "outgoing", "lazy", "thoughtful", "childish" for personalities. 
(e.g. If the turtle is observed to be under the tree over 5 times in recent observations, it would be judged in a "think bad" mood. 
If the turtle is observed to be under the tree or look at the turtle, and these two status present 200 more times than the rest of the status, your turtle would be judged as a "thoughtful" turtle.) 
The current mood, the personality, and the topic you speak would act as the keywords to query from our response database created by MongoDB, and the sentence found would be the turtle's reply to what you talked. 
The response would be read out by the text-to-speech module on your phone. All these are in real-time and runs interactively.



### Results

![image](https://github.com/hz2538/iot_JennyGo/blob/master/raspberry/pic06.jpg)
Status classification results (accuracy and the confusion matrix)

### App

![image](https://github.com/hz2538/iot_JennyGo/blob/master/raspberry/pic09.jpg)

### References


Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

http://icsl.ee.columbia.edu/projects/personal-energy-footprinting/

http://icsl.ee.columbia.edu/projects/seus/

https://edblogs.columbia.edu/eecse4764-001-2016-3/

https://edblogs.columbia.edu/eecse4764-001-2017-3/

http://icsl.ee.columbia.edu/iot-class/2016fall/group7/

http://icsl.ee.columbia.edu/iot-class/2016fall/group8/

					

### Contact Our Team

Jiajing Sun: js5504@columbia.edu

Jiayu Wang: jw3689@columbia.edu

Huixiang Zhuang: hz2538@columbia.edu



Columbia University Department of Electrical Engineering 
http://www.ee.columbia.edu

Class Website:
Columbia University EECS E4764 Fall '19 IoT
https://edblogs.columbia.edu/eecs4764-001-2019-3/

Instructor: Professsor Xiaofan (Fred) Jiang
http://fredjiang.com/
