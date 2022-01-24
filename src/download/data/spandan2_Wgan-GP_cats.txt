# Wgan-GP_cats

This is an implementation of Wasserstein GANs with gradient penalty.<br>
Link to the paper is : https://arxiv.org/pdf/1704.00028.pdf
<br><br><br>

Wasserstein GANs use the Earth mover distance instead of TV or JS divergence or KL divergence.
<br>
The weaker the distance, the better is the convergence of GANs. <br>
The other distances mentioned failed in the case of low dimensional manifolds where the distributions may have very little common projection space. 
<br>
The mathematical details of the advantages of this distance can be read here : https://arxiv.org/pdf/1701.07875.pdf<br>
<br>
<br>
The WGAN paper uses RMSprop for optimization and weight clipping to enforce a Lipschitz condition but in WGAN-GP, gradient penalty enforces the Lipschitz and they succefully trained the model using Adam as discussed in detail in the paper.<br>
## Usage
Any image set of size 64x64 can be put in a folder and placed in the images folder. The noise dimension is set to 100 as suggested in 
the paper but one should feel free to play with the parameters like z_dim, n_critic. Further use of a an optimizer with beta1=0 like RMSprop helps improve results in some cases.
### Results
<b>Epoch 1 </b><br><br>
<img src="sample_images/wgan_gp/Epoch 1.jpg">
<br><br><br><br>
<b>Epoch 100 </b><br><br>
<img src="sample_images/wgan_gp/Epoch 100.jpg">
<br><br><br><br>
<b>Epoch 300</b><br><br>
<img src="sample_images/wgan_gp/Epoch 300.jpg">
<br><br><br><br>
<b>Epoch 500</b><br><br>
<img src="sample_images/wgan_gp/Epoch 500.jpg">
