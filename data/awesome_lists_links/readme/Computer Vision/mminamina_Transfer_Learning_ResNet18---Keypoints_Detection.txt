## TransferLearning ResNet18: Facial Keypoints Detection with ResNet18
Project1: Pretrained ResNet18 in Pytorch for Facial Keypoints Detection 
### Project Experience
This is the version with pretrained ResNet18 for my first computer vision project which allows me to apply deep learning computer vision architectures (using **Pytorch**) to build a facial keypoint detection system for
- tracking
- pose recognition
- filters
- emotion recognition
### Image Augmentation
Introduce generalization(randomness) to detect and learn structures
### Transform Steps
Only the following transformation techniques were chosen to avoid unnecessary impacts on keypoints
- **Rescale** to 250 for width and height
- **RandomCrop** to 224
- **Normalize & Convert ToTensor**

### Architecture & Performance
Pretrained ResNet18 is applied here as TransferLearning. 
Compared to the model without using TransferLearning: https://github.com/mminamina/Udacity_ComputerVision_PeerReviewed_Project---Facial_Keypoints_Detection
Compared to the model with NaimishNet TransferLearning: https://github.com/mminamina/Transfer_Learning_NaimishNet--Keypoints_Detection        

BatchSize, Epochs, Loss & Optimization Functions(using **GPU**)
- **BatchSize** : 32 for train dataset and 20 for test dataset
- **Epochs**   : 8 (can train longer for better performance)
- **Loss**     : SmoothL1Loss (significant loss reduction since earlier epochs)
- **Optimizer** : Adam 

### Worklist
For project instructions, please refer to https://github.com/udacity/P1_Facial_Keypoints

### Files in Order
- models.py;
- Notebook 2. Define the Network Architecture.ipynb
- Notebook 3. Facial Keypoint Detection, Complete Pipeline.ipynb

### Results
- model without TransferLearning vs NaimishNet vs Pretrained ResNet18
- Pretrained ResNet18 produces the best performance and the most accurate keypoints among all three models
- Both NaimishNet and Pretrained ResNet18 generate more accurate keypoints than the simple model without transfer learning

### Additional Helpful Resources
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
- https://arxiv.org/pdf/1512.03385.pdf
- https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035

#### Packages
Python, Pytorch

LICENSE: This project is licensed under the terms of the MIT license.
