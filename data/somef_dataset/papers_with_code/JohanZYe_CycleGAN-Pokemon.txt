# CycleGAN to Swap Pokemon Types
Project conducted with Hyemee Sohn

An implementation of CycleGAN in order to swap Pokemon types.

Original paper: https://arxiv.org/abs/1703.10593

<b> Potential Improvements to Investigate </b>
* Add new different pictures of same pokemon to increase training set
* Implement distill's conv-resample to attempt to remove checkerboard pattern artifact
* Implementation of InstaGAN in order to change geometric shape
* Implement the identity mapping loss [junyanz repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/322).


<b> Observations During Training </b>
* Sometimes identical mapping will be learned after a lot of training
* Sometimes colors are mapped indescriminantly (often blue and red in between fire and water)
* Change in learning rate can cause large changes at times

### Display of Results ###
#### Water to Fire ####
![Water_Fire](Examples/Water_Fire.jpg)

#### Fire to Water ####
![Fire_Water](Examples/Fire_Water.jpg)

#### Grass to Water ####
![Grass_Water](Examples/Grass_Water.jpg)

#### Water to Grass (Note how red and pink barely changes) ####
![Water_Grass](Examples/Water_Grass.jpg)

#### Electric to Water ####
![Electric_Water](Examples/Electric_Water.jpg)

#### Water to Electric ####
![Water_Electric](Examples/Water_Electric.jpg)

#### Electric to Grass ####
![Electric_Grass](Examples/Electric_Grass.jpg)

#### Grass to Electric ####
![Grass_Electric](Examples/Grass_Electric.jpg)


