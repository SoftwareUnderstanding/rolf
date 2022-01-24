# Style Transfer
*by Aryan Mann and Stefan Ilic*
<p align="center">
  <img width="400" height="400" src="https://github.com/aryan-mann/style-transfer/blob/master/images/results/C1S1__done/(99).png">
  <img width="400" height="400" src="https://github.com/aryan-mann/style-transfer/blob/master/images/results/C2S2__done/(99).png">
</p>
<p align="center">
  <img width="400" height="400" src="https://github.com/aryan-mann/style-transfer/blob/master/images/results/C1S2__done/(99).png">
  <img width="400" height="400" src="https://github.com/aryan-mann/style-transfer/blob/master/images/results/C2S1__done/(99).png">
</p>

Any visit to an art museum leaves the observer with a sense of the style of each individual work. This project aims to create a process by which the style of one painting is transferred to another algorithmically. The result is a beautiful fusion between the content and style between two art pieces.

## VGG-19 Architecture

#### Layers of the VGG-19 Image Classification Network
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_conv4 (Conv2D)        (None, None, None, 256)   590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_conv4 (Conv2D)        (None, None, None, 512)   2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0
=================================================================
Total params: 20,024,384
Trainable params: 20,024,384
Non-trainable params: 0
_________________________________________________________________
```

## Image Citations
**Style Images**
- "Starry Night" by Vincent van Gogh, Oil on Canvas. Open Access, url: a.gov/collection/art-object-page.72328.html
- "At the Water's Edge" by Paul Cézanne, Oil on Canvas. Open Access, url: https://www.nga.gov/collection/art-object-page.53119.html

## Citations
- How to pre-process the image to use as input for the VGG-19 network: https://www.programcreek.com/python/example/92217/keras.applications.vgg19.preprocess_input
- How to get the output from an intermediate layer in the VGG using backend functions: https://stackoverflow.com/questions/52230874/how-to-correctly-use-an-intermediate-layer-of-a-vgg-model 
- How to create a new directory: https://stackoverflow.com/questions/1274405/how-to-create-new-folder
## References 
- ***"A Neural Algorithm of Artistic Style"*** by L. Gatys, A. Ecker, and M. Bethge: http://arxiv.org/abs/1508.06576.
- ***"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"*** by J. Johnson, A. Alahi, Li Fei-Fei: https://arxiv.org/abs/1603.08155
- ***"Instance Normalization: The Missing Ingredient for Fast Stylization"*** by D. Ulyanov, A. Vedaldi, V. Lempitsky: https://arxiv.org/abs/1607.08022
- ***"Demystifying Neural Style Transfer"*** by Y. Li, N. Wang, J. Liu, X. Hou
:https://arxiv.org/pdf/1701.01036.pdf
