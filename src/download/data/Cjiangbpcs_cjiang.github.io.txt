# One Solution About Multiple Object Tracking

Multiple Object Tracking, MOT, aims to track motion trajectories of multiple objects in a series of video frames. This task includes two aspects: object detection and data association. Specifically, for each frame, the first step is to detect interested objects, which are then used to track objects by data association. 

Several approaches have been used to tackle MOT task, the following method is relatively mature for production: Backend tracking optimization algorithm based on Kalman Filter and Hungaraian (Kuhn-Munkres) algorithms, with SORT and DEEP-SORT as examples. 

SORT, Simple Online and Realtime Tracking, uses a linear velocity model and Kalman Filter to detect and predict object locations. Hungarian algorithm then serves as a matching optimization algorithm by searching for the maximum matching of a bipartite graph. SORT uses IOU distance, Intersection Over Union distance, as weights to integrate the Hungarian algorithm for data association. SORT also adopts linear assignment implementation directly from sklearn. A threshold of 0.3 for IOU is selected in the original paper to determine whether or not two objects have the same identity. SORT assumes that objects do not move too much between any two frames. This assumption, together with occlusion issues, implies that SORT suffers from high Identity Switches. Its open-sourced version can be found here: https://github.com/abewley/sort.

DEEP-SORT was later developed to integrate SORT's Kalman Filter and data association with appearance information, which successfully reduced 45 percent of SORT's Identity Switches. DEEP-SORT also adds Deep Association Metric, which is mainly used to effectively differentiate two objects, this further improves its accuracy. The architecture can be shown in the following. Its open-sourced version, DEEP-SORT plus YOLO3 (https://arxiv.org/abs/1804.02767), can be found here: https://github.com/nwojke/deep_sort. Another open-sourced version, DEEP-SORT plus SSD512 (https://arxiv.org/abs/1512.02325), can be found here: https://github.com/lyp-deeplearning/deep-sort.

Note that a voting scheme is proposed to be incorporated in DEEP-SORT to identify the best detections among face, object, and Kalman Filter. Together with data association to assign the detections, we expect our solution to be able to provide optimal multiple object tracking, and thus accurate unique customer counting and waiting queue depth. 

![my image](DEEP-SORT.jpg#center)

![my image](IOU_assignment.jpg#center)

![my image](matching_cascade.jpg#center)


# References:
Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich "Simple Online and Realtime Tracking with a Deep Association Metric" In ICIP 2017. https://arxiv.org/abs/1703.07402

Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben "Simple Online and Realtime Tracking" In ICIP 2016. https://arxiv.org/abs/1602.00763

Harold W. Kuhn. The Hungarian Method for the assignment problem. Naval Research Logistics Quarterly, 2:83-97, 1955.

Harold W. Kuhn. Variants of the Hungarian method for assignment problems. Naval Research Logistics Quarterly, 3: 253-258, 1956.

Munkres, J. Algorithms for the Assignment and Transportation Problems. J. SIAM, 5(1):32-38, March, 1957.

https://blog.csdn.net/zjc910997316/article/details/83721573
