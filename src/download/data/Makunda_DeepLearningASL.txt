# Deep Learning ASL Project
# Object Recognition, Tracking and Trajectory prediction 

## Purpose of the study 
The visual tracking of objects is increasingly becoming a very important issue as tracking algorithms have made major advances with Deep Learning algorithms. Today, several methods are available to perform visual tracking tasks and data analysis.

In this project we are moving towards software development, where the goal will be to test the latest deep learning algorithms for tracking objects and calculating their trajectories.

## I.    Detection and Tracking via Deep Learning

<u>Choice of the network :</u>
     Mask R-CNN Facebook AI Research (FAIR) 2018 [https://arxiv.org/pdf/1703.06870.pdf]
    - Excellent accuracy, but long calculation time
    - Trained on a COCO type dataset (Common Object in Context)

We didn't had a lot of "firepower" during this project to run and train our deep learning model. 
We first used a NVDIA GTX 1060, and then a RTX 2070.
So the implementation of a low consuming ressources solution was our priority. We still wanted to use the R-CNN model even if it's not the fastest one, the quality of detection was a this time extremely good. 

<u>There are several solutions to this problem:</u>
 - Tracker implementation : To track object detected in a first pass by the algorithm 
 - Image processing provided to the network : the quality of the image was improved, and most of the background removed 
 - Use of homography and template matching, to repositioned the detected box in a previous DL algorithm run 
 - Program using multi-threading


## II.    Trajectory Prediction via Kalman Filters

//ToWrite

### III.    Trajectory Prediction via Deep Learning

//ToWrite

## Licence 
<i> This project is initally based on the work of Waleed Abdulla. You can find his profile here : https://github.com/waleedka. 
  Some parts of his work were modified and some bugs fixed.
Many thanks to him for the quality of his work, and the clarity of his code.</i>
  
