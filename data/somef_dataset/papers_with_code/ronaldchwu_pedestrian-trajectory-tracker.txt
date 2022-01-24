# Track pedestrian trajectories for space usage planning
<img src="https://github.com/ronaldchwu/pedestrian-trajectory-tracker/blob/main/assets/aws-solution-architecture.png" width="1200">

## Overview
Understanding how people walk around in a space is useful. Knowing how customers explore a retail stores reveals the most or least visited area, so that the store owner can improve store layouts and staff placement. Knowing how pedestrian walk through public spaces helps identify points of congestion and possible barrier of evacuation. Security camera videos provide useful data for such analysis. Computer vision AI made it possible to simply analyze videos and extract pedestrian trajectories, without the need to attach any tracking device to people. Video analytics is projected to increase in market size from $1.1 billion in 2018 to $4.5 billion in 2025 ([report by Tractica](https://omdia.tech.informa.com/OM011985/Video-Analytics)).  With both increased market demands and advances in object-tracking AI algorithms, more and more machine learning solution providers are offering pedestrian tracking services to meet clients' needs.

Interested in how such AI solution is developed, I decided to use open-source software and cloud-computing platforms to build a pedestrian tracker service from scratch. I aim at deploying a easily maintainable and accessible service on AWS cloud. It allows me to minimize running costs, while making use of AWS's MLOps functionality to experiment with various computer vision algorithms.

This service allows users to simply upload a video to a cloud storage (AWS S3), and then receive 1) an annotated video with people in tracking boxes and 2) the detailed trajectory of each person.  The trajectories can be projected to 2D floor plan for detailed spatial flow analyses. All the underlying analyses are automatically triggered by the video upload, and are processed using the serverless AWS Fargate. Data scientists and developers can experiment with different versions of computer vision models and pre- and post-processing scripts (with SageMaker Experiments, Debugger), save them as model checkpoints (in S3) and Docker Image (in AWS ECR), and easily deploy them on the Fargate service.

*-- note 27-Mar-2021: The multi-object tracking models are developed and tested. Trajectory analysis is the next step.*

## Methods
Pedestrian tracking is a multi-object tracking (MOT) problem. It involves detecting people in video frames using deep learning models, and associating the positive detections to specific personID using some tracking algorithms. Therefore, to deliver good solutions, we need to select good combinations of deep learning model and tracking algorithm.

Here I experimented with one simple, baseline solution and one state-of-the-art (SOTA) solution. 

**A) Baseline solution:** ([YOLOv3-SORT.ipynb](YOLOv3-SORT.ipynb))
- Use classic object detection deep learning model (YOLOv3) to detect people in each video frame. This and other classic models are widely available on different frameworks (Tensorflow, PyTorch, mxnet) and can be easily imported and used. Here I use the [gluoncv implementation of the YOLOv3 model](https://cv.gluon.ai/build/examples_detection/demo_yolo.html#sphx-glr-build-examples-detection-demo-yolo-py).
- Use a Simple Online and Realtime Tracking (SORT) algorithm that identify people's trajectories based only on locations of the positive detection bounding boxes. This approach does not require learning about each person's appearance (e.g. color of cloth) and is easy to implement (with just one .py script, using implementation of ([abewley/sort](https://github.com/abewley/sort))). The algorithm offers multiple parameters for fine-tuning; for example, how many consecutive frames do we allow a person to be missing, and the threshold of box overlap for declaring a match. 

**B) SOTA solution:** ([FairMOT.ipynb](FairMOT.ipynb))
- Use FairMOT, a deep learning model specifically designed for multi-object tracking. This deep neural network can simultaneously detect people and learn about their individual feature embeddings (person's appearance). 
- Use a tracking algorithm that uses both locations and feature embeddings to associate positive detections to specific person ID.

The SOTA model has a much more complicated algorithm design. Fortunately, the authors of FairMOT provides open-source implemention scripts of both the detection and tracking tasks ([ifzhang/FairMOT](https://github.com/ifzhang/FairMOT/blob/master/src/track.py)). With a few customized script modification, the model can be run and deployed on AWS cloud environment.

For both solutions, I used pre-trained models to examine their preliminary performance. In production, both models should be trained on proper training data sets.

Below I tested the solutions with a video clip of people in shopping mall ([link](https://www.pexels.com/video/people-walking-inside-a-shopping-mall-4750076/)). It is not seen by the models before, and present some occlusion challenges.

## Performance
### Baseline solution: YOLOv3 + SORT
<img src="assets/shopping-mall2-SORT-results-largefont.gif" width="600"/> 

We can see that the baseline model does not detect people well, especially those at the far end of the floor. More critically, occlusion seems to be causing big problems. People who walk in groups sometimes got their ID swapped. It is not surprising, because the YOLOv3 model used here is the 'vanilla' version based on ResNet-34 backbone, with no configuration for tackling occlusion, scaling (people further away is smaller) and deformation (people may change posture). Also, the SORT algorithm is prone to mis-identification when people form groups and walk pass each other.

### SOTA solution: FairMOT
<img src="assets/shopping-mall2-results-FairMOT-ct03dt03-largefont.gif" width="600"/> 

The FairMOT model does a much better job. Occlusion problem is reasonably resolved. Although sometimes the same person is 'flashing' and keep being assigned new ID, it is much easier to fix on trajectory maps compared to the swapping problem above. 

There are at least three reasons for the model's good performance: 1) it uses a deformable convolution network with deep layer aggregation backbone (DLA-34), which handles scaling and deformation better than YOLOv3-ResNet34; 2) explicit learning of each person's appearance greatly improved re-identification during tracking; 3) the model was pre-trained on multi-object detection datasets, allowing effective application into our use case. 

It is fascinating that the FairMOT model does quite well on a video clip it never saw before. In practice, a better performance can be expected after formally trained and fine-tuned the model on training data set of the same camera. Our MLOps architecture makes it easier to do so, and to deploy improved model checkpoints into production.

## Next to implement
* Project pedestrian trajectories onto 2D floor plans.
* Post-process trajectories (e.g. fix inaccurate re-id over frames)

## Acknowledgements
Thanks to the developers of SORT and FairMOT for providing open-source implemention scripts. 
For details of FairMOT model, please refer to the original publication:
> [**FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**](http://arxiv.org/abs/2004.01888),            
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu,        
> *arXiv technical report ([arXiv 2004.01888](http://arxiv.org/abs/2004.01888))*
