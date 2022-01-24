# Change My Pet
<div align="center">
    <a>
        <img src="https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/blob/master/assets/logo1.png?raw=true0" width="300" height="120">
    </a>
</div>
<br />

> ChangeMyPet provides a deep learning model that is capable of replacing a real dog/cat image with a GAN(Generative adversarial network) generated image with the help of segmentation masks.

# Results:
<div align="center">
    <a>
        <img src="https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/blob/master/assets/example2.png?raw=true" />
    </a>
</div>
<hr />
<div align="center">
    <a>
        <img src="https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/blob/master/assets/example1.png?raw=true">
    </a>
</div>
<br />


# Exciting Gifs:
<div align="center">
    <a>
        <img src="https://raw.githubusercontent.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/master/assets/gif2.gif" width="500" height="500"/>
    </a>
</div>
<br />
<div align="center">
    <a>
        <img src="https://raw.githubusercontent.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model/master/assets/gif1.gif" width="500" height="500"/>
    </a>
</div>

<br />

The results are quite interesting and the model is learning to generate images in the required segment. Regularization and ensembling losses have given more accurate results but we are still exploring other techniques. Our next step is to include Discriminator so that the images generated look more real. Generate a pull request if you want to conrtribute to the project.

# Demo

- Clone the repository in your GitHub account [Clone with HTTPS](https://github.com/harpreetsodhi/ChangeMyPet_Deep_Learning_Model.git)
- To run the code, please download the pretrained pytorch weights first. [Pretrained Weights](https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases/tag/v0.0.0)
```shell
    biggan256-release.pt    # download this for generating 256*256 images
```
- Upload the biggan256-release.pt file to your google drive.
- Open Main.ipynb file in Google Colab or your Jupyter Notebook and run it. Comments are added to the file as needed.

# References 
paper: https://arxiv.org/abs/1809.11096

https://github.com/ivclab/BIGGAN-Generator-Pretrained-Pytorch

https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
