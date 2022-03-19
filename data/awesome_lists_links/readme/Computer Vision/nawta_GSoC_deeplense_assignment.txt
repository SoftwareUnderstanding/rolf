# GSoC_deeplense_assignment


## TASK 1
Q.Modify and/or use the already existing functionality of PyAutoLens to simulate strong lensing from superfluid dark matter. Specifically, you will need to simulate the effects of lensing from a linear mass density - imagine this being a string of mass on galactic scales.

A.First of all, For simplicity, I am going to think the dark matter discussed in this task as axion. This is because axion is a representative of the dark matter with superfluidity, and is in the paper of DEEPLENSE project[1] .

In this task,
I will start with consideration from a linear mass density.  
According to pp.3[1], vortex solution is characterized by a density profile that can be parameterized below

![](https://user-images.githubusercontent.com/39507181/77626407-ac967b80-6f88-11ea-8aa7-3bdfe15afcd2.png)

This models the vortex as a tube. Therefore, we can use line integral to it and the vortex's mass can be calculated.

there're two ways in the aspect of implementation.  
First, I implement the dark matter halo as a spherical isothermal, like the related work[1] does. I use the parameter according to the paper.  
Second, I implement it using mass-sheet class, which is at PyAutoAstro project[2].  
In this case, according to [3], I calculate kappa(The magnitude of the convergence of the mass-sheet) in the equation below.   
(x: the distance from spectator to source plane, or lensed galaxy)  
(k: kappa)  
  
k(x) = 1/(2x)  
  
I implement in both ways. (still working on the second method)
  
 ------ 
 
### Task1 Reference:  
   
 [1] Deep Learning the Morphology of Dark Matter Substructure
 https://arxiv.org/abs/1909.07346
 
 [2] https://github.com/Jammy2211/PyAutoAstro/blob/master/autoastro/profiles/mass_profiles/mass_sheets.py
   
 [3] Introduction to Gravitational Lensing Lecture scripts, pp37,39,41
 http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf
 
 ------
 
 
## TASK 2

Q. Using a deep learning algorithm of your choice, learn the representation of dark matter in the strong gravitational lensing images provided using PyTorch.
The goal is to use this learned representation for anomaly detection, hence you should pick the most appropriate algorithm and discuss your strategy.   
   
A. Considering the purpose is anomaly detection, I choose Autoencoder as the most appropriate method, as discussed in [1].  
I think Conditional VAE (CVAE)[5] especially suits this purpose.   
Deep neural network architecture is known to function relatively better in tasks about images than MLP(multilayer perception), and in the early stage Convolutional AutoEncoder(hereafter CAE)[2] and Variational AutoEncoder(VAE)[8] were the most competent method.
Of cource, it is said that CNN architecture in general performs well, speaking of image processing.  
CAE incorpolates CNN's algorithm into itself, and according to [3],[4] (written in Japnanese), it clearly shows that CAE overwhelms other MLP algorithms.   
VAE incorpolates probability distribution into latent variable z, and it also performs as well as CAE does.
These days, lots of methods derived from these, and I noticed CVAE is the best, which add supervising features to VAE.  
it is compared with lots of other autoencoder methods on purpose of analysis on astronomical images[6], and prove that the latent variables obtained with CVAE preserve more information and are more discriminative towards real astronomical transients.(SCAE[7] did well as well, but slightly CVAE has an advantage of it)    
This paper was published in April 2018, and it is the most recent paper applying autoencoder to processing astronomical images.(According to my survey on Google Scholor)  
In addition to that, CVAE is implementable on laptop, whereas other methods sometimes need a bulk of computer properties(such as [9].)  
  
Using such autoencoders, anomaly detection will be easy.(in future work I can impliment heat map which clearly tells anomalies from norms [10] )  
__I couldn't do operational check because of my laptop's lack in spec__
  
According to Hold-out method, I devided images into train(3001 images) and test(2000 images).   
Directory structures are below. you should put lenses directory at the same place as this notebook.   
- lenses/train/sub/(3001 images)
- lenses/train/no_sub/(3001 images)
- lenses/test/sub/(2000 images)
- lenses/test/no_sub/(2000 images)
  
About optimizer, I use Adam because this is most popular optimizers' father and since this is not that complicated, the convergence speed is fast.  
  
in inplementation, I refered to this site:   
https://graviraja.github.io/conditionalvae/#  
  
  
  
Of cource, CNN could be used because this problem can be taken as a classification problem(so no_sub images provided)  
~~In case, I also implement CNN, which was written at the end of this notebook.(I used ResNet, as in [11])~~
I couldn't implement CNN case in time.  
  
  
### Task2 References:  
  
----  
  
[1]Deep Learning the Morphology of Dark Matter Substructure, pp8  
https://arxiv.org/pdf/1909.07346.pdf

[2]Deep Clustering with Convolutional Autoencoders  
https://link.springer.com/chapter/10.1007/978-3-319-70096-0_39

[3]PytorchによるAutoEncoder Familyの実装(Implementation of various Autoencoders using PyTorch)  
https://elix-tech.github.io/ja/2016/07/17/autoencoder.html

[4]様々なオートエンコーダによる異常検知(Anomaly Detection with lots of Autoencoders)  
https://sinyblog.com/deaplearning/auto_encoder_001/

[5]Semi-Supervised Learning with Deep Generative Models (CVAE)  
https://arxiv.org/abs/1406.5298

[6]Latent representations of transient candidates from an astronomical image difference pipeline using Variational Autoencoders  
https://pdfs.semanticscholar.org/57ce/a9520008706b8cb190d94513e695068210cd.pdf

[7]Convolutional Sparse Autoencoders for Image Classification (SCAE)  
https://ieeexplore.ieee.org/document/7962256

[8]Auto-Encoding Variational Bayes(VAE)  
https://arxiv.org/pdf/1312.6114.pdf

[9]Improving Variational Autoencoder with Deep Feature Consistent and Generative Adversarial Training (GAN)  
https://arxiv.org/pdf/1906.01984.pdf

[10]Anomaly Manufacturing Product Detection using Unregularized Anomaly Score on Deep Generative Models  
https://confit.atlas.jp/guide/event-img/jsai2018/2A1-03/public/pdf?type=in  
https://qiita.com/shinmura0/items/811d01384e20bfd1e035

[11] Deep Learning the Morphology of Dark Matter Substructure https://arxiv.org/abs/1909.07346
  
  
