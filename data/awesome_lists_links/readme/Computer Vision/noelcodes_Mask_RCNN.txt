# [Home Page](https://noelcodes.github.io/)
# Mask_RCNN 
Curious what is Mask_RCNN, so I followed tutorial by Mark Jay and made it working on my laptop.
So it is actually Faster RCNN (object detection with bounding boxes) + Masks

#### Reference
Paper by the Facebook AI Research (FAIR) : https://arxiv.org/pdf/1703.06870.pdf <br>
Theory: https://www.youtube.com/watch?v=g7z4mkfRjI4 <br>
Tutorial : https://www.youtube.com/playlist?list=PLX-LrBk6h3wRAF22jBUxDgOvyhIgLN4Cg <br>
Repo: https://github.com/matterport/Mask_RCNN <br>
Repo: https://github.com/markjay4k/Mask-RCNN-series <br>

Run on my laptop, see my demo video below:
[![LIVE DEMO](https://github.com/noelcodes/Mask_RCNN/blob/master/ezgif.com-video-to-gif.gif)](https://youtu.be/_I44RHIIe2A) 

![alt text](https://i.imgur.com/u1TkBuf.jpg)   

#### Installation
Step 1: create a conda virtual environment with python 3.6
```
conda create -n MaskRCNN python=3.6 pip
```
Step 2: install the dependencies
```
pip install -r requirements.txt
```

Step 3: Clone the Mask_RCNN repo
```
git clone https://github.com/matterport/Mask_RCNN.git
```
Step 4: install pycocotools
NOTE: pycocotools requires Visual C++ 2015 Build Tools
```
git clone https://github.com/philferriere/cocoapi.git
use pip to install pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
Step 5: download the pre-trained weights
Go here https://github.com/matterport/Mask_RCNN/releases
download the mask_rcnn_coco.h5 file
place the file in the Mask_RCNN directory

#### Test on static images
open up the demo.ipynb and run it.
On static images it works pretty well.

![alt text](https://i.imgur.com/5gxZwCw.png)   
![alt text](https://i.imgur.com/IbU73eX.png)   
![alt text](https://i.imgur.com/4WhUnhf.png)   

#### Test it on webcam
Credits to Mark Jay's youtube tutorial. 
Goto sample folder, Run "python visualize_cv2.py" in terminal.
Stream on webcam is quite lagging. I think only 2fps. I read that the creator only got 5fps. I'm not sure if I use a light model will help or not. But I guess I can't change it since Mask is built on Faster-RCNN.
Click below for my video demo.

 


