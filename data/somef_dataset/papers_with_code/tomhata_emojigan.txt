# EmojiGAN
Generate emojis with a DCGAN
<br>
<img src="assets/emoji_gan_banner.png" />

# Overview
EmojiGAN is a Deep Convolutional Generative Adversarial Network that generates emojis. A perfect companion for AI twitterbots. This model was trained on a GTX 1080ti using Keras with Tensorflow backend.

## Project Requirements
* Download the "emoji_imgs_V5" data set at https://github.com/SHITianhao/emoji-dataset
* Dependencies:
	* Tensorflow (with GPU support if you plan to train the model)
	* Keras
	* OpenCV

## Resources and acknowledgements
* CatDCGAN by Thomas Simonini https://github.com/simoninithomas/CatDCGAN/
* Deep Learning With Python by Francois Chollet https://github.com/fchollet/deep-learning-with-python-notebooks
* Original paper on DCGAN by Radford and Metz https://arxiv.org/pdf/1511.06434.pdf

# Instructions

 - Move downloaded emoji files into directory **'/data/emoji_test_set/'**.
 - (Optional) Remove any unwanted groups of emojis. E.g., the data used in the model located in **'/models/'** was trained with approximately half the data set omitted. The omitted data were inanimate objects such as flags, buildings, etc. A text file list of the emojis used in the training set can be found in the **'/models/'** directory.
 - Run the **emojigan.ipynb** Jupyter notebook file for an interactive walkthrough to generate and train the model. 
 - Trained models are available in the **'/models/'** directory. The generator model can be loaded and used to create new emojis.

## Coming soon

 - Plain python script equivalent of **emoji_dcgan.ipynb** to generate and train the DCGAN model.
 - Script to automatically generate new emojis using the trained generator model.

 <img src="assets/emoji_gan_progression.gif" />
