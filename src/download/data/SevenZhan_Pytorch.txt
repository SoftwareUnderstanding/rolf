### Tips:
  1. **TORCH_HOME:** used to custom model dir when downloading pretrained models from model zoos;

### Datasets:
  1. **Split2TnV:** used to create train and validation set in the form of txt file;
  2. **ItemList:** taken the txt files create from Split2TnV and used to create pytorch datasets which then can be load with pytorch dataloader;


### Losses:
  1. **SVSoftmax:** Support Vector Guided Softmax Loss for Face Recognition: https://arxiv.org/pdf/1812.11317.pdf;
  2. **Arcface:** ArcFace: Additive Angular Margin Loss for Deep Face Recognition: https://arxiv.org/pdf/1801.07698.pdf;


### Schedulers:
  1. **CyclicalLR:** Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/pdf/1506.01186.pdf;
  2. **CosineLR:** SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS: https://arxiv.org/pdf/1608.03983.pdf;


---
### CosineAnnealingWarmRestarts(1.3.0):
  1. **T_0:** cycle length;
  2. **T_mult:** cycle length multiplier;
![cosine_annealing_warm_restarts](https://user-images.githubusercontent.com/20135989/68026469-cccc5200-fcea-11e9-9399-11e4a5a5eae3.png)

### OneCycleLR(1.3.0):
  1. **max_lr**: maximum learning rate value;
  2. **total_steps:** cycle length, can also be set by using _epochs_ & _steps_per_epoch_;
![one_cycle](https://user-images.githubusercontent.com/20135989/68027019-40229380-fcec-11e9-884b-087e22adf7a3.png)