# Final project for Computer Vision Lab with Solomon Jacobs

Task: Create a neural network to perform tasks relevant to the vision system for a soccer playing robot.
 - Detect objects in the image: soccer balls, other robots and goal posts
 - Segment the image into: field, lines and background
 
We used a U-net like network based on the paper [1]. A network based on DeepLabV3 [2] was also tested.

More details can be found in our report.

## Visualization
Our results on the detction task are visualised here. The middle line is the ground truth and the bottom line is our prediction. We learned heatmaps to predict the centerpoints of objects. Green represents goalposts, Blue represents other robots and red represents football balls.
![](/images/detection_task.png?)


The next image depicts the results on the segmentation task.
![](/images/segmentation_task.png)


## Citations

1 Farazi, H. et al. (2018.) NimbRo Robots Winning RoboCup 2018 Humanoid AdultSize Soccer Competitions. Retrieved from http://arxiv.org/abs/1909.02385 

2 Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. Retrieved from http://arxiv.org/abs/1706.05587
