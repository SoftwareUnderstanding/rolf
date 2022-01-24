# Stochastic-constant-pruninglayers
[PyTorch] experiment CIFAR-10 dataset with ResNet stochastic-model and constant-model pruning layers 

ablation study case:
ablation study studies the performance of an AI system by removing certain components, to understand the contribution of the component to the overall system

in this tutorial authors trying to prove that transfer learning capability of in stochastic model is more superior than constant in ResNet architecture Case<br/><br/>

### <b>1. Original Model architecture of ResNet50 : </b> <br/><br/>
<img src="https://github.com/Anderies/stochastic-constant-pruninglayers/blob/master/Figure/CONV_BEFORE_1.png" width="50%" height="50%">

### <b>2. Modified Model architecture with pruning layers: </b> <br/><br/>
<img src="https://github.com/Anderies/stochastic-constant-pruninglayers/blob/master/Figure/CONV-AFTER-1.png" width="50%" height="50%">

## Experiment Result 

in this experiment authors is trying to remove sequential layer in sequence to make model leaner.

### <b>3. Overall experiment result in ResNet50 stochastic depth: </b><br/>
<img src="https://github.com/Anderies/stochastic-constant-pruninglayers/blob/master/Figure/stochastic%20experiment.png" width="50%" height="50%">

### <b>4. Overall experiment result in ResNet50 constant* depth: </b><br/>
<img src="https://github.com/Anderies/stochastic-constant-pruninglayers/blob/master/Figure/constant%20experiment.png" width="50%" height="50%">

*constant is standard resnet50 architecture by Kaiming He et al. in Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385 <br/>
*stochastic depth is resnet50 architecture by Gao Huang et al. in Deep Networks with Stochastic Depth https://arxiv.org/abs/1603.09382
