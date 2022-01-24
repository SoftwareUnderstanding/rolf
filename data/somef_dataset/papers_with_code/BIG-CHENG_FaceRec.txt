# FaceRec

簡單易懂，高精準度的人臉辨識技術封裝


# Papers

深度學習人臉辨識技術

0. 基礎: 機器學習/深度學習/圖形處理器技術

1. "DeepFace: Closing the Gap to Human-Level Performance in Face Verification"

*https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf?spm=5176.100239.blogcont55892.18.pm8zm1&file=Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf

最早的深度學習人臉辨識, 已有 metric learning 的觀念 (使用 siamese network)

但, 無權值共享的 CNN 帶來過多的參數, 3D alignment 也顯得過度複雜


2. "Deep Face Recognition" 

*http://cis.csuohio.edu/~sschung/CIS660/DeepFaceRecognition_parkhi15.pdf

著名的 VGG Face, 整套流程包含 face dataset 的建立


3. "FaceNet: A Unified Embedding for Face Recognition and Clustering"

*https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf

用 triplet loss 產生 128 維的 FaceNet embeddings (此向量空間內的距離代表人臉的相似程度), LFW 準確度超過 99%

網路結構:

101. (A) "Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations"

*https://arxiv.org/pdf/1409.1556/

經典的 VGG Network, 包含 VGG16, VGG19

102. "Going Deeper With Convolutions"

http://openaccess.thecvf.com/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf

GoogLeNet, 使用 3x3, 1x1 convolution 構成 inception 網路模組

103. "Deep residual learning for image recognition"

http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

residual network, 解決梯度消失問題, 讓訓練 100 (甚至1000) 層以上的深度學習變得容易

104. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

https://arxiv.org/abs/1704.04861

mobile net, 小而快的網路， 但犧牲準確度， 

A. "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments"

http://cs.brown.edu/courses/cs143/2011/proj4/papers/lfw.pdf

*著名的 lfw 人臉辨識準確率測試資料集

# Results
99%
*https://github.com/BIG-CHENG/FaceRec/blob/master/fr_lfw_prec_recall_all.png
![LFW precision-recall ](https://github.com/BIG-CHENG/FaceRec/blob/master/fr_lfw_prec_recall_all.png)
*https://github.com/BIG-CHENG/FaceRec/blob/master/fr_lfw_roc_all.png
![LFW ROC ](https://github.com/BIG-CHENG/FaceRec/blob/master/fr_lfw_roc_all.png)

