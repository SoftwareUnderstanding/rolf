# SemanticSegmentation
Semantic segmentation on aerial images (aka image classification) using a CNN-based UNet.

## Implementation
- trains from scratch a CNN with a UNet architecture with 3 (or 4) lateral connections: https://arxiv.org/pdf/1505.04597.pdf
- valid-padding is used to avoid border issues
- segmentation maps for each image patch are stitched back together to create complete map

## Data
- 9 cm spatial resolution aerial images (3-channel: NIR-Red-Green) of the town of Vaihingen from the ISPRS 2D Semantic Labeling Contest: 
http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html 
- corresponding normalized Digital Surface Model (nDSM) from:
https://www.researchgate.net/publication/270104315_Normalized_DSM_-_heights_encoded_in_dm_-_see_report_for_details
- we use NIR, Red and nDSM (discarding Green), as Keras' ImageDataGenerator accepts only 3 channels

## Results
- after a few epochs of training on a laptop GPU: OA: 83.33, Kappa: 0.778 (on separate test set)
- misses classes "cars" and "clutter/background" (small nr of samples): stratified sampling to be implemented to solve the issue

Ground Truth               |  Prediction
:-------------------------:|:-------------------------:
![](figures/GT_map.png?raw=true "Ground Truth")  |  ![](figures/predicted_map.png?raw=true "Prediction")