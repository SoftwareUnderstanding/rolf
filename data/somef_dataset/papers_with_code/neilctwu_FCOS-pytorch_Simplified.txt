# Simplified FCOS (pytorch)
Re-implementation of FCOS: Fully Convolutional One-Stage Object Detection (https://arxiv.org/abs/1904.01355)

---
#### Modified codes from:  
https://github.com/rosinality/fcos-pytorch

##### What's different?
- Simplified as I can to increase legibility  
- Implementing in-house (tooth detection) data loader instead of COCO dataset.
- Removing parallel computing functions
---
### Backbone
Using [Vovnet57](https://arxiv.org/abs/1904.09730) as backbone
Pre-trained pth download from [here](https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1)([ref](https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py)) 

---
##### TO-DO:  
- Add modified **evalute funtion** for in-house data  
- Add **image output function** for visualizations
