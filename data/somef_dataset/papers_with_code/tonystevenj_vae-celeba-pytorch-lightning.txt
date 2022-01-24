# Vanilla Variational Auto Encoder
Vanilla VAE implemented in pytorch-lightning, trained through Celeba dataset. The mse loss used is 'sum' instead of 'mean'. 


Need further optimization, but for now, we can see the result of sampling is close to training result. Which tells the theory works pretty good.


  - View the code there: https://github.com/tonystevenj/vae-celeba-pytorch-lightning/blob/main/2020-fall-ml.ipynb

    - Note: This modle require long training time, and I trained it on Amazon SageMaker Studio. If you prefer Colab, pls don't forget to change root folder path to your mounted Google Drive, or you will lose your process after 12 hours (Longest Colab session is 12 hours).

    - you can set your root folder path by setting "default_root_dir='/content/drive/MyDrive'" to Trainner. [See official ducoment here](https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html).

  - Link to the paper: https://arxiv.org/abs/1312.6114

  - Link to Celeba dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# Train

<div align=center>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/t1.png"/>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/t2.png"/>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/t3.png"/>
</div>


# Sampling


<div align=center>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/s1.png"/>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/s2.png"/>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/s3.png"/>
<img src="https://raw.githubusercontent.com/tonystevenj/vae-celeba-pytorch-lightning/main/s4.png"/>
</div>


