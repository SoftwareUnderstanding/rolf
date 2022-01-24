# Group 2 - Ethnicity Cycle GAN

**Used articles and papers**
1. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
Implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
Paper: https://arxiv.org/pdf/1703.10593.pdf

<img src="https://cdn-images-1.medium.com/max/800/1*nKe_kwZoefrELGHh06sbuw.jpeg" width=500>


2. Article having (1.) as basis

https://hackernoon.com/gender-and-race-change-on-your-selfie-with-neural-nets-9a9a1c9c5c16

**Goal**  
Having Cycle GANs for ethnicity transformation, e.g.  
1. *Black and White*  
<img src="https://cdn-images-1.medium.com/max/800/1*yFZY_gIOXP5Squmq0TBItA.png" width=600>
 
 
1. *White and Asian*  
<img src="https://cdn-images-1.medium.com/max/800/1*3ihWND1xfqTNP_uEgZviYw.png" width=600>


**Used Datasets**  
1. CelebA ~ 200.000 images  
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  
<img src="http://mmlab.ie.cuhk.edu.hk/projects/celeba/intro.png" width=400>

2. UTKFace ~ 20.000 images  
http://aicip.eecs.utk.edu/wiki/UTKFace  
<img src="http://aicip.eecs.utk.edu/mediawiki/images/thumb/e/ef/LogoFaceWall2.jpg/700px-LogoFaceWall2.jpg" width=400>
3. LFW (Labeled Faces in the Wild) Database  ~ 13.000 images
http://vis-www.cs.umass.edu/lfw/  

**Faced Problems**  
1. CelebA is a huge dataset but does not have ethnicity labels unfortunately.  
*Approach: train Classifier on UTKFace and LFW to label CelebA.*  

**Improvements**
2. Replace L1 pixel to pixel identity loss with p2 Norm of feature map distance.
see https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf