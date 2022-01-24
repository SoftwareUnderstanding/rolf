# VQ-VAE
Implementation of paper: Neural Discrete Representation Learning (VQ-VAE)
</br >
Original Paper: https://arxiv.org/pdf/1711.00937.pdf
</br ></br >
Note: This implementation is done only on CIFAR-10 dataset.
</br ></br >
VQ-VAE.ipynb file contains the training and inference of VQVAE model along with the training and inference of pixelCNN model trained on top of discrete latents produced in previous step.
</br ></br >
To train the pixelcnn model, first VQVAE model has to be trained. We are using pretrained model weights of VQVAE model which is also provided in this repository as 'vqvae_25000.pth' file.
</br ></br >
If only need to produce the outputs, the pretrained weights are given (refer the given notebook file for visualizing). Also if need to produce the random outputs using pixelcnn 
prior, again weights are given "vqvae_pixelcnn_prior_65.pth" (refer the given notebook file for visualizing).
</br ></br >
Reconstruction of images using VQVAE model.
</br >
Image on left is a recontruction for original image on right.
</br >
![vqvae_reconstruction](https://user-images.githubusercontent.com/39181807/120303427-72430f80-c2ec-11eb-952b-6a9872cc3df1.png)
![vqvae_original](https://user-images.githubusercontent.com/39181807/120303443-7838f080-c2ec-11eb-9d48-ef24174c488f.png)
</br ></br >
Random images generated using the pixelCNN prior.
</br >
![sample1](https://user-images.githubusercontent.com/39181807/120303839-d82f9700-c2ec-11eb-8306-e43e27cf2810.png)
