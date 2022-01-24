# Imbalanced Data
Battling the unblananced dataset problem using different data augmentation methods

The network models in the project use the area under the ROC curve (AUC)[1] as a metric for assessing prediction performance. Overall accuracy is not a suitable metric as it does not give a good overview of seperation power for unbalanced datasets[2]. AUC on the other hand uses recall and precision, meaning it takes advantage of the confusion matrix[2] of the model and will thus give a more suitable measurement for models working on imbalanced datasets.

Model Selection (Cross Validation) using AutoKeras[5] and some popular network models - Best performer: LENET 300
- [X] Random Undersampling
- [X] Oversampling through standard duplication
- [X] Oversampling through duplication with small noise
- [X] Oversampling using SMOTE [3]
- [ ] Oversampling using mixup [4]



References
1. Andrew P. Bradley - 'The Use of the Area Under the ROC Curve in The Evaluation of Machine Learning Algorithms' - https://linkinghub.elsevier.com/retrieve/pii/S0031320396001422
2. Sofia Visa, Ramsay Brian, Ralescu Anca - 'Confusion Matrix-based Feature Selection' - http://ceur-ws.org/Vol-710/paper37.pdf
3. Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer - 'SMOTE: Synthentic Minority Over-sampling Technique' - https://arxiv.org/pdf/1106.1813.pdf
4. Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz - 'mixup: Beyond Empirical Rsik Minimization' - http://arxiv.org/abs/1710.09412
5. Jin, Haifeng and Song, Qingquan and Hu, Xia - Auto-Keras: An Efficient Neural Architecture Search System - https://dl.acm.org/doi/10.1145/3292500.3330648
