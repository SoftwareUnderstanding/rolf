Shake-Shake Regularization
=====

TensorFlow implementation of Shake-Shake Regularization.  

## Concept
<div align="center">
  <img src="./figures/shake.png" width="600">  
  <p>The concept of Shake-Shake Regularization [1].</p>
</div>

## Procedure

The whole procedure for using Shake-Shake Regularization is shown as below. All the figures are redesigned by <a href="https://github.com/YeongHyeon">YeongHyeon</a>.  

<div align="center">
  <img src="./figures/phase0.png" width="600">  
  <p>Phase 0. Preparing for Shake-Shake.</p>
</div>

<div align="center">
  <img src="./figures/phase1.png" width="600">  
  <p>Phase 1. Forward propagation in training.</p>
</div>

<div align="center">
  <img src="./figures/phase2.png" width="600">  
  <p>Phase 2. Backward propagation in training.</p>
</div>

<div align="center">
  <img src="./figures/phase3.png" width="600">  
  <p>Phase 3. Forward propagation in test.</p>
</div>

## Performance

The performance is measured using below two CNN architectures.

<div align="center">
  <img src="./figures/cnn.png" width="600">  
  <p>Two Convolutional Neural Networks for experiment.</p>
</div>

| |ConvNet8|ConvNet8 with S-S|
|:---|:---:|:---:|
|Accuracy|0.99340|0.99420|
|Precision|0.99339|0.99416|
|Recall|0.99329|0.99413|
|F1-Score|0.99334|0.99414|

### ConvNet8
```
Confusion Matrix
[[ 979    0    0    0    0    0    0    1    0    0]
 [   0 1132    0    1    0    0    1    1    0    0]
 [   0    0 1029    0    0    0    0    3    0    0]
 [   0    0    1 1006    0    3    0    0    0    0]
 [   0    0    1    0  975    0    2    0    0    4]
 [   1    0    0    7    0  882    1    0    0    1]
 [   4    2    0    0    0    1  950    0    1    0]
 [   1    3    3    2    0    0    0 1018    1    0]
 [   3    0    1    1    0    1    0    0  966    2]
 [   0    0    0    1    6    2    0    3    0  997]]
Class-0 | Precision: 0.99089, Recall: 0.99898, F1-Score: 0.99492
Class-1 | Precision: 0.99560, Recall: 0.99736, F1-Score: 0.99648
Class-2 | Precision: 0.99420, Recall: 0.99709, F1-Score: 0.99565
Class-3 | Precision: 0.98821, Recall: 0.99604, F1-Score: 0.99211
Class-4 | Precision: 0.99388, Recall: 0.99287, F1-Score: 0.99338
Class-5 | Precision: 0.99213, Recall: 0.98879, F1-Score: 0.99045
Class-6 | Precision: 0.99581, Recall: 0.99165, F1-Score: 0.99372
Class-7 | Precision: 0.99220, Recall: 0.99027, F1-Score: 0.99124
Class-8 | Precision: 0.99793, Recall: 0.99179, F1-Score: 0.99485
Class-9 | Precision: 0.99303, Recall: 0.98811, F1-Score: 0.99056

Total | Accuracy: 0.99340, Precision: 0.99339, Recall: 0.99329, F1-Score: 0.99334
```

### ConvNet8 with S-S (ConvNet8 + Shake-Shake Regularization)
```
Confusion Matrix
[[ 978    1    0    0    0    0    0    1    0    0]
 [   0 1131    0    0    0    0    2    1    1    0]
 [   1    1 1027    0    0    0    0    2    1    0]
 [   0    0    0 1008    0    2    0    0    0    0]
 [   0    0    0    0  979    0    1    0    0    2]
 [   1    0    0    6    0  884    1    0    0    0]
 [   3    2    0    0    2    1  948    0    2    0]
 [   0    1    4    0    1    0    0 1020    1    1]
 [   2    0    2    0    0    1    0    0  967    2]
 [   0    0    0    0    4    3    0    1    1 1000]]
Class-0 | Precision: 0.99289, Recall: 0.99796, F1-Score: 0.99542
Class-1 | Precision: 0.99560, Recall: 0.99648, F1-Score: 0.99604
Class-2 | Precision: 0.99419, Recall: 0.99516, F1-Score: 0.99467
Class-3 | Precision: 0.99408, Recall: 0.99802, F1-Score: 0.99605
Class-4 | Precision: 0.99290, Recall: 0.99695, F1-Score: 0.99492
Class-5 | Precision: 0.99214, Recall: 0.99103, F1-Score: 0.99159
Class-6 | Precision: 0.99580, Recall: 0.98956, F1-Score: 0.99267
Class-7 | Precision: 0.99512, Recall: 0.99222, F1-Score: 0.99367
Class-8 | Precision: 0.99383, Recall: 0.99281, F1-Score: 0.99332
Class-9 | Precision: 0.99502, Recall: 0.99108, F1-Score: 0.99305

Total | Accuracy: 0.99420, Precision: 0.99416, Recall: 0.99413, F1-Score: 0.99414
```

## Requirements
* Python 3.6.8  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  

## Reference
[1] Gastaldi, Xavier. <a href="https://arxiv.org/abs/1705.07485">Shake-Shake Regularization.</a> arXiv preprint arXiv:1705.07485 (2017).
