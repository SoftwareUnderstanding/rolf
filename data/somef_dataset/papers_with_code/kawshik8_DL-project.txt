# Bird's Eye View layout prediction: roads and cars
#### Final project for Deep Learning course (DS-GA 1008, NYU Center for Data Science)  
#### Top-10 overall rank in road layout prediction and car bounding boxes prediction tasks
#### Kawshik Kannan, Hsin-Rung Chou, Dipika Rajesh

#### [Report](DL_final_report.pdf) | [Video](add link for video)

## Abstract
In this project we focus on Bird's Eye View (BEV) prediction based on monocular photos taken by the cameras on top of the car. We experiment with Determinisitic autoencoders, stochastic variational autoencoders, generative adversarial networks for generating Bird's eye view road layout and Bird's eye view of vehicles on the road indirectly. THe best performing models on the training set use GANs whereas the maximum test performance was from the deterministic model. Our models achieve **0.904 val threat score on the road layout prediction task and 0.044 val threat score on the BB prediction task**. 

---
## Usage
### Generate and save labels
Use `generate_labels.py` to generate
- vehicles mask
- road mask
- warped and glued photos


### Road Layout Prediction and Bounding Boxes Prediction
Refer to `src/` for code used to train and test road layout prediction models. 
- GANs `src/GANmodels`<br>
- Deterministic models and Retinanet `src/SupModels`<br>
- training and validation scripts `src/trainer` <br>
- training and validation scripts `src/trainer` <br>


### Self-supervised learning 
Implemented PIRL and SIMCLR SSL techniques in src/SSLmodels.py


---
Libraries used<br>
- [Pytorch](https://pytorch.org/docs/stable/index.html)
- [OpenCV](https://opencv.org/)

### Papers and useful links:
- simclr https://arxiv.org/abs/2002.05709 <br>
- PIRL https://arxiv.org/abs/1912.01991
- retinanet https://arxiv.org/abs/1708.02002 <br>
- rotation based object detection https://arxiv.org/pdf/1911.08299.pdf






