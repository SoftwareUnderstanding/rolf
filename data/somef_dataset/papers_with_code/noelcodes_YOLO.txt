# [Home Page](https://noelcodes.github.io/)

# Exploring YOLO. 
Just wanted to check out what Yolo is, so followed tutorial by Mark Jay, and made it working on my laptop.

Yolo paper: https://arxiv.org/abs/1506.02640 <br>
Tutorial Video: https://www.youtube.com/playlist?list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM <br>
Darkflow Repo: https://github.com/thtrieu/darkflow <br>
YOLO weights: https://pjreddie.com/darknet/yolo/ <br>

[![LIVE DEMO](https://github.com/noelcodes/YOLO/blob/master/ezgif.com-video-to-gif%20(2).gif)](https://youtu.be/pTLJpy_TIZo)

### YOLO (You Only Look Once)
Reference : https://github.com/noelcodes/Capstone-Project/blob/master/yolo_noel/darkflow/yolo_noel.py

Let's have a gimps of Yolo. What a name!! 
For details you can read the paper here https://arxiv.org/abs/1506.02640. 
Yolo is very fast, outperformed any other method we have discussed, mainly because it detects the object location in a smarter way. 
The general idea is: 
- The input image is divided into an S × S grid. 
- For each grid cell predicts B bounding boxes, confidence for those boxes and C class probabilities.
- These predictions are encoded as an S x S x (B*5+C) tensor.
- Finally set a threhold level of 30% of confidence prediction, this removes all low confidence level boxes, leaving only the high predicted ones.

![alt text](https://i.imgur.com/EuMMiub.jpg)   

#### Installation
Yolo is written in Darknet, Open Source Neural Networks in C, for its speed.
Click here for the link to [Darknet](https://pjreddie.com/darknet/yolo/).

But we are died hard python fan. Luckily some one translated darknet to a Tensorflow version, hence the name darkflow. You can clone it from this [Darkflow link](https://github.com/thtrieu/darkflow).
Follow thru the installation guide written there. Basically it uses Cython language to bridge between C to Python. Then run my code 'yolo_noel.py" in terminal. 
Download you own weight too, as filesize too large, I am unable to push it up to my github.

##### Custom images + Demo
I did not have time to use my custom image on this, preparation of the labelmaps and xml format is slightly different from the tensorflow method. I also noticed that the accuracy of Yolo model is not as good as Faster R-CNN, so I did not bother to waste time on it. However, it is still possible to use custom images. Instruction is on the same link above. 

For advertising tagging purpose, we just need to replace labels with resepctive slogan/promotions. Example:
```
label = result['label']
if label == 'laptop':
     label='Laptop now $999'
elif label == 'Wine':
     label='Buy 1 Wine, Get 1 Free'
```     

![alt text](https://i.imgur.com/F0GhB4D.jpg)
