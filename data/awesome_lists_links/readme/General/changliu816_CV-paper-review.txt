# Category

|:dog:|:mouse:|:hamster:|:tiger:|
|------|------|------|------|
|[1.Domain Adaptation](#1)|[2.Domain Generalization](#2)|[3.Image Self-supervised Learning](#3)|[4.Video Self-supervised Learning](#4)|
|[5.Semi-supervised Learning](#5)|[6.Long-tailed Recognition](#6)|[7.Noisy-label learning](#7)|[8.Data Augmentation](#8)|
|[9.Few-shots/Meta Learning](#9)|[10.Metric Learning](#10)|[11.Adversarial Learning](#11)|[12.Image2Image Translation](#12)|
|[13.Transformer/self-attention](#13)|[0.Classification/Fine-grained](#0)|


# Survey
* [Transfer learning](https://github.com/yuntaodu/Transfer-learning-materials)
* [Self-superivsed learning](https://github.com/jason718/awesome-self-supervised-learning) 
* [Domain generalization](https://github.com/amber0309/Domain-generalization)
* [semi-supervised learning](https://github.com/yassouali/awesome-semi-supervised-learning)

<a name="1"/>

## 1.Domain Adaptation
* [χ-MODEL: IMPROVING DATA EFFICIENCY IN DEEP LEARNING WITH A MINIMAX MODEL, ICLR2022](https://arxiv.org/pdf/2110.04572.pdf) 

* [Self-Tuning for Data-Efficient Deep Learning, ICML2021](https://arxiv.org/pdf/2102.12903.pdf) [[code]](https://github.com/virajprabhu/CLUE)

* [Active Domain Adaptation via Clustering UNcertainty-weighted Embeddings, ICCV2021](https://arxiv.org/pdf/2010.08666.pdf) [[code]](https://github.com/virajprabhu/CLUE)

* [Code: Transfer learning library from Tsinghua Pytorch](https://github.com/thuml/Transfer-Learning-Library)

* [On Generating Transferable Targeted Perturbations, ICCV2021](https://arxiv.org/pdf/2103.14641.pdf) [[code]](https://github.com/Muzammal-Naseer/TTP)

* [Adversarial Unsupervised Domain Adaptation with Conditional and Label Shift: Infer, Align and Iterate, ICCV2021](https://arxiv.org/abs/2107.13469)


* [SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation, ICCV2021](https://arxiv.org/abs/2012.11460) 


* :star:[Gradient Distribution Alignment Certificates Better Adversarial Domain
Adaptation, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Gradient_Distribution_Alignment_Certificates_Better_Adversarial_Domain_Adaptation_ICCV_2021_paper.pdf) [code](https://github.com/theNded/SGP)
   * Improvement for MCD third step: Fix two classifier and update G by minimize divegence between prob output of two classifiers.
   * The new divergence loss is: gradience discrepancy: Cosine-distance (gs,gt) where gs is the grad. of L_ce and gt is the weighted pseudo label loss. 
   * New way to select PL: calculate target proto by weighted features where weights are from the softmax outputs. 

* [Adaptive Adversarial Network for Source-free Domain Adaptation, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf) 

* [STEM: An approach to Multi-source Domain Adaptation with Guarantees, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Nguyen_STEM_An_Approach_to_Multi-Source_Domain_Adaptation_With_Guarantees_ICCV_2021_paper.pdf) 

* [Dynamic Transfer for Multi-Source Domain Adaptation, CVPR2021](https://arxiv.org/pdf/2103.10583.pdf) 


* :star:[Transferable Semantic Augmentation for Domain Adaptation, CVPR2021](https://arxiv.org/pdf/2103.12562.pdf) [[code]](https://github.com/BIT-DA/TSA)





<a name="2"/>

## 2.Domain Generalization


* [VICREG: VARIANCE-INVARIANCE-COVARIANCE REGULARIZATION FOR SELF-SUPERVISED LEARNING, ICLR2022](https://arxiv.org/pdf/2105.04906.pdf) 
     *  用作DG 


* [Quantifying and Improving Transferability in Domain Generalization, NIPS2021](https://arxiv.org/pdf/2106.03632.pdf) [[code]](https://github.com/virajprabhu/CLUE)


* [Confidence Calibration for Domain Generalization under Covariate Shift, ICCV2021](https://arxiv.org/abs/2104.00742) 
* :star:[Domain Generalization via Gradient Surgery](https://arxiv.org/abs/2108.01621)[[code]](https://github.com/lucasmansilla/DGvGS)
  * Motivation: Multi-task learning has gradients conflict issues; This paper hypothesizes that the gradients conflict also exsits in domains.
  * Method: update CNN by harmonizing inter-domain gradients. 判断不同source的gradients的sign是否一致。如果一致，就keep，然后average;如果不一致，可以zero or assign a random value to them. 
  * **Comments**不同domains or tasks可能会有gradients conflict这个角度很有趣。可以拓展到few-shots或者 有很多loss的时候，怎么确保彼此是否confilct.
  * **Extension** 是否可以把不同source的gradient投射到某一个axis，而不是直接zero。 
* [Deep Domain-Adversarial Image Generation for Domain Generalisation, AAAI2020](https://arxiv.org/pdf/2003.06054.pdf) [[code]](https://github.com/KaiyangZhou/Dassl.pytorch)


<a name="3"/>

## 3.Image Self-supervised Learning


* :star:[THE CLOSE RELATIONSHIP BETWEEN CONTRASTIVE LEARNING AND META-LEARNING,ICLR2022](https://openreview.net/pdf?id=gKLAAfiytI)[[code]](https://github.com/UMBCvision/MSF)
   * 两种形式： 一个是把large rotation/mix其他一些augmentation 当成新的sample instance class, 然后放入contrastive; 或者加入rotation prediction loss. 后者效果显著。

* :star:[EQUIVARIANT SELF-SUPERVISED LEARNING: ENCOURAGING EQUIVARIANCE IN REPRESENTATIONS,ICLR2022](https://openreview.net/pdf?id=gKLAAfiytI)[[code]](https://github.com/UMBCvision/MSF)
    * 解释了，一些augmentation 适合learn invariance，做contrastive learning; 一些augmnetation, e.g rotation 适合learnequivariance, 做prediction loss; 两者是complemantary,结合起来效果更好。


* :star:[Masked Autoencoders Are Scalable Vision Learners,arxiv](https://arxiv.org/pdf/2111.06377.pdf)[[code]](https://github.com/UMBCvision/MSF)


* :star:[Mean Shift for Self-Supervised Learning,ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Koohpayegani_Mean_Shift_for_Self-Supervised_Learning_ICCV_2021_paper.pdf)[[code]](https://github.com/UMBCvision/MSF)

* [A Broad Study on the Transferability of Visual Representations With Contrastive Learning](https://arxiv.org/abs/2103.13517)<br>:star:[code](https://github.com/asrafulashiq/transfer_broad)
* [With a Little Help From My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations](https://arxiv.org/abs/2104.14548)
* :star:[On Feature Decorrelation in Self-Supervised Learning, ICCV2021](https://arxiv.org/abs/2105.00470) 

* [Improving Contrastive Learning by Visualizing Feature Transformation](https://arxiv.org/abs/2108.02982)<br>:open_mouth:oral:star:[code](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)

* [ISD: Self-Supervised Learning by Iterative Similarity Distillation, ICCV2021](https://arxiv.org/pdf/2012.09259.pdf) [code](https://github.com/UMBCvision/ISD)

* [Switchable K-class Hyperplanes for Noise-Robust Representation Learning, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Switchable_K-Class_Hyperplanes_for_Noise-Robust_Representation_Learning_ICCV_2021_paper.pdf) [code](https://github.com/liubx07/SKH.git)

* [Solving Inefficiency of Self-supervised Representation Learning, ICCV2021](https://arxiv.org/pdf/2104.08760.pdf) [code](https://github.com/wanggrun/triplet)

* :star:[Understanding Self-Supervised Learning Dynamics without Contrastive Pairs,ICML2021](http://proceedings.mlr.press/v139/tian21a/tian21a.pdf)
* :star:[toward Understanding the Feature Learning Process of Self-supervised
Contrastive Learning,ICML2021](http://proceedings.mlr.press/v139/wen21c/wen21c.pdf)


* [Self-supervised Motion Learning from Static Images, CVPR2021](https://arxiv.org/pdf/2104.00240.pdf)

* [Joint Contrastive Learning with Infinite Possibilities, NIPS2020](https://arxiv.org/pdf/2009.14776.pdf) [[code]](https://github.com/caiqi/Joint-Contrastive-Learning)
    * Extension for Moco: pushes the num of positives to infinity and minimizes the upper bound of loss. E(log(X))<= log E(X)
    * Key is to estimate the Gaussian distribution of positive keys N(mu, sigma)
    * **Comments**: 类似 Semantic Aug，只是loss 从CE 换成了Contrastive Loss. 

* [Contrastive Learning with Adversarial Examples, NIPS2020](https://arxiv.org/pdf/2010.12050.pdf) [code](https://github.com/chihhuiho/CLAE)
    * 2-steps optimization: 1) Generate positive adv samples by maximizing CTloss. 2) Use Adv samples for CTLoss.

* [Adversarial Self-Supervised Contrastive Learning, NIPS2020](https://arxiv.org/pdf/2006.07589.pdf) [code](https://github.com/Kim-Minseon/RoCL)
    * **Comments:和上面那篇基本一样，但是citation更高

* [Towards Domain-Agnostic Contrastive Learning, ICML2021](http://proceedings.mlr.press/v139/verma21a/verma21a.pdf)

* [i-MIX: A DOMAIN-AGNOSTIC STRATEGY FOR CONTRASTIVE REPRESENTATION LEARNING, ICLR2021](https://openreview.net/pdf?id=T6AxtOaWydQ) [[code]](https://github.com/kibok90/imix)


<a name="4"/>

## 4.Video Self-supervised Learning
* [Contrast and Order Representations for Video Self-Supervised Learning, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Contrast_and_Order_Representations_for_Video_Self-Supervised_Learning_ICCV_2021_paper.pdf)
* [Contrastive Learning of Image Representations With Cross-Video Cycle-Consistency, ICCV2021](https://arxiv.org/abs/2105.06463)<br>:house:[project](https://happywu.github.io/cycle_contrast_video/)
 * [Composable Augmentation Encoding for Video Representation Learning, ICCV2021](https://arxiv.org/abs/2104.00616)
 * [Motion-Focused Contrastive Learning of Video Representations, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Motion-Focused_Contrastive_Learning_of_Video_Representations_ICCV_2021_paper.pdf)
 * [Time-Equivariant Contrastive Video Representation Learning, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jenni_Time-Equivariant_Contrastive_Video_Representation_Learning_ICCV_2021_paper.pdf)
  *  [Spatiotemporal Contrastive Video Representation Learning,CVPR21](https://arxiv.org/abs/2008.03800)<br>:star:[code](https://github.com/tensorflow/models/tree/master/official/) 
 *  [Removing the Background by Adding the Background: Towards Background Robust Self-Supervised Video Representation Learning,CVPR21](https://arxiv.org/abs/2009.05769) 
 *  [VideoMoCo: Contrastive Video Representation Learning withTemporally Adversarial Examples,CVPR21](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_VideoMoCo_Contrastive_Video_Representation_Learning_With_Temporally_Adversarial_Examples_CVPR_2021_paper.pdf) 
       * Two improvements upon Moco: 1) generate temporal adversarial example as positives. 2) temporal decay to reduce the contributions from old keys.  

<a name="5"/>

## 5.Semi-supervised Learning

* [ON NON-RANDOM MISSING LABELS IN SEMI-SUPERVISED LEARNING, ICLR2022](https://openreview.net/pdf?id=6yVvwR9H9Oj)
  
  
  
* [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling, NIPS2021](https://arxiv.org/pdf/2110.08263.pdf)
  
 * [Trash to Treasure: Harvesting OOD Data with Cross-Modal Matching for Open-Set Semi-Supervised Learning](https://arxiv.org/abs/2108.05617)

  * [Semi-Supervised Active Learning for Semi-Supervised Models: Exploit Adversarial Examples With Graph-Based Virtual Labels](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Semi-Supervised_Active_Learning_for_Semi-Supervised_Models_Exploit_Adversarial_Examples_With_ICCV_2021_paper.pdf)
  * [CoMatch: Semi-Supervised Learning With Contrastive Graph Regularization](https://arxiv.org/abs/2011.11183)<br>:star:[code](https://github.com/salesforce/CoMatch)
  * 
* [Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples, ICCV2021](https://arxiv.org/pdf/2104.13963.pdf) [code](https://github.com/facebookresearch/suncet)

* [Weakly Supervised Contrastive Learning, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Weakly_Supervised_Contrastive_Learning_ICCV_2021_paper.pdf) [code](https://github.com/theNded/SGP)

* [Weakly Supervised Representation Learning with Coarse Labels, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Weakly_Supervised_Representation_Learning_With_Coarse_Labels_ICCV_2021_paper.pdf) [code](https://github.com/theNded/SGP)
* [Semi-Supervised Action Recognition With Temporal Contrastive Learning](https://arxiv.org/abs/2102.02751)<br>:star:[code](https://github.com/CVIR/TCL)

<a name="6"/>

## 6.Long-tailed Recognition

<a name="7"/>

## 7.Noisy-label learning


 
 <a name="8"/>

## 8.Data Augmentation
* [On Feature Normalization and Data Augmentation, CVPR2021](https://arxiv.org/pdf/2002.11102.pdf) [code](https://github.com/Boyiliee/MoEx.)
    * ConvNets trained on ImageNet are biased towards textures instead of shapes. The moments (mean, variance) contains rich structure information and should not be discarded by normalization. 
    * Swap the shape (or style) information of two images by swapping the moments after 1st layers. 
    * Feature-level augmentation. 类似 cross-norm那篇paper.

* [SuperMix: Supervising the Mixing Data Augmentation, CVPR2021](https://arxiv.org/pdf/2002.11102.pdf) [code](https://github.com/Boyiliee/MoEx.)
    * ConvNets trained on ImageNet are biased towards textures instead of shapes. The moments (mean, variance) contains rich structure information and should not be discarded by normalization. 
    * Swap the shape (or style) information of two images by swapping the moments after 1st layers. 
    * Feature-level augmentation

* :star:[Implicit Semantic Data Augmentation for Deep Networks, NIPS2019](https://papers.nips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf) [[code]](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks)



<a name="9"/>

## 9.Few-shots/Meta Learning



 <a name="10"/>
 
 ## 10.Metric Learning
* [Towards Interpretable Deep Metric Learning with Structural Matching](https://arxiv.org/abs/2108.05889)<br>:star:[code](https://github.com/wl-zhao/DIML)
* [Deep Relational Metric Learning](https://arxiv.org/abs/2108.10026)<br>:star:[code](https://github.com/zbr17/DRML)
* [LoOp: Looking for Optimal Hard Negative Embeddings for Deep Metric Learning](https://arxiv.org/abs/2108.09335)<br>:star:[code](https://github.com/puneesh00/LoOp)


<a name="11"/>

## 11.Adversarial Learning
* :star:[TRANSFERABLE PERTURBATIONS OF DEEP FEATURE DISTRIBUTIONS, ICLR2020](https://arxiv.org/pdf/2004.12519.pdf)
  * Motivation:当用混淆class prediction作为white-box model的criterion时，遇到class不一致的blackbox model, adv examples很难transfer
  * Method: 用intermidiate feature 后面跟一个auxillary classifier作为criterion，让adv example混淆它。这样transfer效果更好。加大混淆力度：model误以为是其他class，model prediction原理src class； 生成的adv feature与orgional feature拉远。
 
* :star:[Adversarial Examples Improve Image Recognition, CVPR2020](https://arxiv.org/pdf/1911.09665.pdf)[[code]](https://github.com/tingxueronghua/pytorch-classification-advprop)
  
* [TOWARDS FEATURE SPACE ADVERSARIAL ATTACK, AAAI2021](https://arxiv.org/pdf/2004.12385.pdf)
* [Feature Space Perturbations Yield More Transferable Adversarial Examples, cvpr2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Inkawhich_Feature_Space_Perturbations_Yield_More_Transferable_Adversarial_Examples_CVPR_2019_paper.pdf)




<a name="12"/>

## 12.Image2Image Translation


<a name="13"/>

## 13.Transformer 



<a name="0"/>

## 0. Classification/Fine-grained
[Towards Learning Spatially Discriminative Feature Representations, ICCV2021](https://arxiv.org/pdf/2109.01359.pdf)<br>:open_mouth:oral:star:[code](xxx)
* propose CAM-loss to push CAAM and CAM to be close by l1 distance. 
* **Comments**: 1.CAAMs代表所有objects出现的地方的attention, CAM代表某个特定object的attention. 一般来说，非target objects会干扰分类，所以让CAAM去拉近CAM可以避免这个问题。2. 可以拓展到Knowledge distillation. 用teacher的CAM 去guide student的CAAMs.

