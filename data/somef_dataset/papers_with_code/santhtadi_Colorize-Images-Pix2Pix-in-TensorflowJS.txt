# Colorize-Images-Pix2Pix-in-TensorflowJS

## Live Demo
A Live demo can be found at [santhtadi.github.io](https://santhtadi.github.io/Colorize-Images-Pix2Pix-in-TensorflowJS/)

# Credits
Huge thanks to https://github.com/yining1023/pix2pix_tensorflowjs_lite and 
https://github.com/affinelayer/pix2pix-tensorflow repos for the detailed explanation.

Please check them out for training your own GAN models.

The dataset used for the project is  **ffhq dataset** available at https://github.com/NVlabs/ffhq-dataset

It is available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license

**Dataset paper citation:**

>A Style-Based Generator Architecture for Generative Adversarial Networks
Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)
https://arxiv.org/abs/1812.04948

# Introduction
This project show cases the use of Tensorflow JS, a JavaScript library released by Tensorflow, for Image Conversion using GANs (Pix2Pix architecture here).

## Advantages
Tfjs addressed the most common problem with deployment of DL models - Setting up the environment.

With tfjs the model outputs can be shown right in the browser, making it available and easier to use for a larger demographic.

Leveraging the tfjs script we can run inferences on the client-side with virtually no setup.

## Disadvantages
Inconsistent User Experience (fps, internet speed) can become a problem for systems with various configurations, but that's the case with all websites and browser apps.

# Steps to use this Repo to generate custom pix2pix model and run inference

1. Find the ipynb Jupyter notebook file in the repo.
2. Upload it to colab and select the GPU instance. Please note that colab uses TF2.0, but affinelayer's repo is built using tfv1, the appropriate changes are suggested in the notebook.
3. Create your own dataset (each image is 256x256) and stitch them side by side (512x256) (Please refer to the repos in credits for more info).
4. Follow the instructions in the notebook to generate the pix2pix model and the tfjs model.
5. Place the pict file in models folder.
6. Edit the index.js file as required.


