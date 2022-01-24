# GAN_Waifu_Generator
A Generative Adversarial Network trained on a dataset of over 60,000 anime girl images to generate new ones. Based on the original model proposed by Goodfellow et al. in the paper "Generative Adversarial Networks", this model simultaneously trains both a generator to create new images and a discriminator to distinguish between the generated images and the real images. Subsequently, the generator then is able to create accurate waifu images which, if you use them as profile pictures as often as I do, is a big help.

## Dataset

Unfortunately I couldn't find the link to the original source of the 60,000 waifu images (;-;). However, the dataset is essentially comprised of 63633 square images of anime girl faces of different resolutions. All of them were resized to be 64x64 when training the model. 

## Sample Results

### Single Images
<p float="left">
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/generated_1_21.png" height="64" width="64">
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/generated_1_20.png" height="64" width="64">
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/generated_1_18.png" height="64" width="64">
</p>

### Groups of Images
<p float="left">
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/generated_103.png" height="384" width="384">
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/generated_104.png" height="384" width="384">
</p>

### Gif 
Note: This is a size-reduced version of the original GIF, hence its lowered resolution and slow change speed.
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/training_visual_downsized.gif" height="384" width="384">

### Training Results
Loss values were calculated as an mean average over the first and last 1000 values for the initial losses and final losses, respectively.
<img src="https://github.com/Chubbyman2/GAN_Waifu_Generator/blob/main/results/loss_values.png">

Initial Generator Loss: 1.02

Final Generator Loss: 1.16

Initial Discriminator Loss: 0.65

Final Discriminator Loss: 0.66

## Future Improvements
This project was mostly limited due to the complexity of the GAN (the GAN used was fairly simple) and the low-quality images used. In the future, if a dataset comprised of higher-resolution waifu pictures was used, it would eliminate some of the effects related to the weird green pixels generated and produce much more usable profile pictures. Also, further research needs to be done into why the loss values for the generator and discriminator did not decrease.

## Acknowledgments
Generative Adversarial Networks by Goodfellow et al.: https://arxiv.org/pdf/1406.2661.pdf

Training code based off of Nagesh Singh Chauhan's: https://www.kdnuggets.com/2020/03/generate-realistic-human-face-using-gan.html
