# Model performance and interpretability of semi-supervised generative adversarial networks to predict oncogenic variants with unlabeled data

In this pilot study, we present a semi-supervised learning model to solve the classification problem with insufficient label data. The result indicates that the predictive performance is slightly improved compared to the existing softwares using 4,000 labeled data and 60,000 unlabeled data. 


## What and Why?

Our semi-supervised model is based on unsupervised **G**enerative **A**dversarial **N**etworks (GANs). Goodfellow et. al firstly proposed [GAN](https://arxiv.org/abs/1406.2661). GANs contain 2 parts: (1) generator and (2) discriminator. In a standard training process, the generator generated fake samples from noise vectors, and then, the discriminator is to distinguish whether the input sample is fake or real. The generator is supposed to fool the discriminator, which means the generator is trained to fit the underlying probability density function of real data. 

<img src="https://github.com/WGLab/SGAN/blob/main/figs/semi-supervised.png" width="300" alt="distribution"/><br/>

In our study, we can assume that the input data (interpretation scores from different guidelines and softwares) follows an unknown distribution (grep points). Then, the discriminator is to classify the data into 3 parts: benign (red point), oncogenic (blue point), and fake. Once the generator has been trained to generate synthtic samples following the underlying distribution, the discriminator can tell us the border of benign/oncogenic class with a small number of labeled data. The detail of our model is shown following:
<img src="https://github.com/WGLab/SGAN/blob/main/figs/ourmodel.png" width="600" alt="distribution"/><br/>

## Run the code

We used conda to build environment and the code is implemented in Pytorch. You can train a model on jupyter notebook.



## Reference:

[1] https://github.com/etaoxing/semi-supervised-gan

[2] https://github.com/opetrova/SemiSupervisedPytorchGAN

[3] https://arxiv.org/abs/1606.03498
