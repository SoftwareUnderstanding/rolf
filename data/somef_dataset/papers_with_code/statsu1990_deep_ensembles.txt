# deep_ensembles
Uncertainty estimation in deep learning using deep ensembles with keras<br>

"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"<br>
https://arxiv.org/abs/1612.01474

In this sample, estimate uncertainty in CNN classification of dogs and cats images using deep ensembles. This CNN predict label which indicates dog 1 or cat 0.<br>
Some of the results are shown below.<br>

![mrc](https://github.com/statsu1990/deep_ensembles/blob/master/result/hist_unc_another.png)<br>
      Fig. The histgram of uncertainty (standart deviation of label).<br>
![mrc](https://github.com/statsu1990/deep_ensembles/blob/master/result/unc_vs_prob_dog.png)<br>
      Fig. The standart deviation of label vs label in dog images.<br>
![mrc](https://github.com/statsu1990/deep_ensembles/blob/master/result/unc_vs_prob_cat.png)<br>
      Fig. The standart deviation of label vs label in cat images.<br>

The details are described in the blog below.<br>
https://st1990.hatenablog.com/entry/2019/08/15/200842

#### classification_uncertainty_score.py.py
- estimate uncertainty in classification of dogs and cats.<br>

#### classifier_cnn.py
- Create binary calassification CNN model predcting label and variance.<br>

#### cifar10_data.py
- Treat cifar10 data.<br>



# deep_ensembles
deepアンサンブルを使ったdeep learningの不確かさ評価(keras)。<br>

"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"<br>
https://arxiv.org/abs/1612.01474

このサンブルでは、犬猫画像のCNNでの分類の不確かさを評価する。CNNは犬1、猫0のラベルを推定する。<br>
詳細は以下のブログ参照。<br>
https://st1990.hatenablog.com/entry/2019/08/15/200842

#### classification_uncertainty_score.py.py
- 犬猫分類の不確かさ評価。<br>

#### classifier_cnn.py
- ラベルと分散を推定する二値分類CNN。<br>

#### cifar10_data.py
- cifar10のデータクラス。<br>
