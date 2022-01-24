# Deep-Zooming-of-Images using SRGAN

Description:
 The folder contain two Python scripts
 1. Zoom.py : Takes jpg images from Input folder,slice the input image into hundred pieces and convert each sliced images into high resolution images using SRGAN pretrained models. 
    Library used : OpenCV, tensorflow
    Python Libs included : glob, numpy, image_slicer, os, math, numpy

 2. join_images.py : Join each high resolution slices and reconstruct the zoomed form of orginal image.
    Library used : OpenCV
    Python Libs included : numpy

The folder contain one .sh command language interprete
 1. Zoom.sh : Contains parameters given to Zoom.py

Developed Based On :
 1. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network-  https://arxiv.org/pdf/1609.04802.pdf
 2. The code is highly inspired by  : pix2pix-tensorflow .
 3. Pretrained model : https://drive.google.com/uc?id=0BxRIhBA0x8lHNDJFVjJEQnZtcmc&export=download 

Dependency:
 1. python2.7 or python 3.6
 2. tensorflow r1.10 or above version
 3. OpenCV 4.0.1 or above

How To Run Code:
 1. Download the zip file Deep-Zooming-of-Images using SRGAN.zip and extract the files in it.
 2. Download the pretrained model from the link given above.
 3. After extracting the folder copy the input image to Input Folder(in jpg format only).
 4. Run the Zoom.sh.
 5. The Output image will be generated in Zoom folder.

Warning: Do not forget to remove the temporary folder (temp) generated, if the programe stops in between execution. If it is'nt removed it may affect next execution.
