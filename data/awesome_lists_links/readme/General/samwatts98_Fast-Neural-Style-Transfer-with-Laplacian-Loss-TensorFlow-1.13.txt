
# About #
This codebase was developed for a final year project in my Computer Science & Artificial Intelligence degree.

Neural Style Transfer is a popular branch of visual machine learning. It is the process of allowing a neural network model to learn the artistic features present within one style target image, and using these features to re-style any other input image whilst still containing the content features present.
### Content Target ###
<img src=".demoimages/brighton.JPG" width="400" />

### Style Targets ###
<img src="https://www.vincentvangogh.org/images/paintings/the-starry-night.jpg" width="400" />
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1920px-Tsunami_by_hokusai_19th_century.jpg" width="400" />
<img src="https://www.pablo-ruiz-picasso.net/images/works/1906.jpg" width="400" />

### Results - all 3 extensions in use ###
<img src=".demoimages/brighton-starry.JPG" width="400" />
<img src=".demoimages/brighton-wave.JPG" width="400" />
<img src=".demoimages/brighton-muse.jpg" width="400" />

The purpose of this project is to provide an interactive user-friendly notebook for users of all programming competency to engage with. The notebook clearly lays out to the user the steps in using the Notebook. Feel free to fork the repository and implement your own published improvements! 

The main file is a Python Notebook called 'StyleTransferSystem.ipynb'.
The helper python '.py' files are also needed in the same directory as these notebooks. 

# Improvements Implemented #
This fast neural style transfer model is based upon the architecture of Johnson et al (2016), available at: (https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

This training system includes several published improvements to the field of neural style transfer:

- Instance normalisation over batch normalisation. Ulyanov et al (2017), available at : (https://arxiv.org/pdf/1607.08022.pdf)
- Laplacian loss for preserving smaller details. Li et al, (2017), available at: (https://arxiv.org/pdf/1707.01253.pdf)
- Resize ppscaling layers over deconvolution layers. Odena et al, (2017), available at: (https://distill.pub/2016/deconv-checkerboard/)

# Using Pre-Trained Style Networks #
Contained in the 'MODELS' directory is a number of example pre-trained networks that can be loaded within the Notebook file.

# License #
Copyright (c) 2015-2019 Sam Watts. Released under GPLv3. See LICENSE.txt for details.

A special thank you to Anish Athalye for their implementation of the VGG-19 network to be used in this project.
Thier source-code is available at: https://github.com/anishathalye/neural-style
