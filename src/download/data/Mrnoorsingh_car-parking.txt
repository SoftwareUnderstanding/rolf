# Car Parking Detection

## Overview

The project uses MRCNN model pre-trained on MS-COCO dataset to detect vehicles and OPENCV library to process(read & write) real time video frames.
To enable SMS alert, Twilio API is utilized to send the SMS on the user's phone

The problem for parking detection can be break down as follows

![pipeline](/img/pipeline.png)
 
 ---
 
 ## Example

  Recorded Video(Output)   |      SMS Received
:-----------------:|:-----------------------:
![recorded video](/img/park.gif) | ![received SMS](/img/Screenshot%20from%202019-08-07%2002-07-53.png)



## References

The model's code and pretrained weights can be downloaded from [here](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py)

https://arxiv.org/pdf/1703.06870

https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400
