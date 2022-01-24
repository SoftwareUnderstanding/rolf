# Estimating_Distance
This app is used for estimating the distance between two balls from a monocular RGB image. 

# DEMO
Upload an image from the left bar, and the result will be output automatically.  
Click the check box (Result semantic segmentation), and you can see reslut of semantic segmentation.
![demo_img1](https://user-images.githubusercontent.com/64745286/119369441-24784700-bcef-11eb-982e-e8d61d497b01.png)


# Features
There are no restrictions on the direction of shooting.
Since real-time semantic segmentation(ESPNetv2) is used in this application, it can be reasoned with fast speed.  

Detail algorizm of this App is here:  
https://www.slideshare.net/RidgeiSlideshare/may-internship-challenge-estimating-distance-between-two-balls-app<br><br>
ESPNetv2 paper link:  
https://arxiv.org/abs/1811.11431<br><br>
Repositry of ESPNetv2 is here:  
https://github.com/sacmehta/EdgeNets


# Confirmed environment
* Python==3.6.9
* torch==1.4.0
* torchvision==0.5.0  
* streamlit==0.48.1
* opencv-python==3.4.1.15 


# Usage
```bash
git clone https://github.com/oki5656/Estimating_Distance.git
streamlit run app.py
```

# Note
The app needs prior information below  
・camera rangle of view  
・number of pixels when the ball is at the reference distance  


# Author
* ikuto oki
* Tsukuba University
