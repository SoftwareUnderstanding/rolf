# Human_Pose_Estimation
Human Pose Estimation project using modified version of Stacked Hourglass architecture proposed by Newell et al. 
(available at https://arxiv.org/abs/1603.06937 )

This repository includes preprocessing (seperate for MPII and UP14 datasets), postprocessing, model and evaluation functions. 

The original Stacked Hourglass architecture is slightly modified in this project;
  - Instead of residual blocks, inception - resnet modules are used. 
  - The loss function is slightly modified. 

# Pretrained Model
A pretrained model using MPII dataset is available at https://drive.google.com/open?id=1sAAI2HX2HjdhbatD0bSy_fFt4Nh7zozI . 





