# Handling imbalanced classification problems with SMOTE

![text](figs/performance.png)

There are several ways of dealing with imbalanced data:

- Undersampling
- Oversampling
- Getting more data
- Use proper weights in the cost function, give more weight to the underrepresented class.

Oversampling technique is better than the other two approaches, it is less expensive than getting more data, andwe do not throw away instances from data that may have some useful information. In this repo I am testing SMOTE (Synthetic Minority Over-sampling Technique) - an oversampling method for generating synthetic samples, this method instead of duplicating samples creates synthetic samples that are interpolations of the minority class [1].
 
__SMOTE__ Steps:

- Randomly select a sample from the minority class 
- Find K nearest neighbours (typically, 5) for that sample
- N out of K samples are selected for the interpolation
- Compute the difference between the selected sample and a neighbour
- Compute gap: select a random number from [0,1]
- Multiply the diffference by the gap and add to the pprevious feature

<p align="center">
<img src="figs/fig3.png", width=500>
</p>

Figure [source](https://github.com/minoue-xx/Oversampling-Imbalanced-Data).

The main disadvantage of this method is that some of the the synthetic features are linearly correlated with each other and with some of the original samples.

There is a variation of the SMOTE algorithm called __ADASYN__. ADASYN is an Adaptive Synthetic Sampling Approach - is a modified version of the SMOTE algorithm [2]. The difference is that ADASYN takes into consideration the density distribution for every minority sample, this distribution affects the number of synthetic samples generated for samples that are difficult to learn. This helps to adaptively change the decision boundary based on the samples difficult to learn.

<p align="center">
<img src="figs/fig4.png", width=500>
</p>

Figure [source](https://github.com/minoue-xx/Oversampling-Imbalanced-Data).


### Note about cross-validation

A brute force oversampling the minority class can result in overfitting if we oversample before cross-validating, since we are using the same data for training and validation (see figs below, [source](https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation))

<p align="center">
<img src="figs/fig1.jpeg", width=300>
<img src="figs/fig2.jpeg", width=300>
</p>

If we first start cross-validatiing, at each iteration we exclude the samples to be used in the validation set, and then oversample the remaining of the minority class.

## References

[1] SMOTE: Synthetic Minority Over-sampling Technique, Nitesh V. Chawla et al, (2002) https://arxiv.org/pdf/1106.1813.pdf

[2] ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced
Learning, H.He et al. (2008) https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf