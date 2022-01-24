# Animation-standing-segmentation

# unet_colorization_pytorch

A segmentation application made by U-Net. It can be used to reconstruction alpha channel.
About the paper:
https://arxiv.org/abs/1505.04597  

For runing this program, you need:  
Pytorch: 1.1.0 or newer  

Result:

Illustrations  -->  Masks(The alpha channel)  -->  Generated

.<img src="https://github.com/zampie/Animation-standing-segmentation/blob/master/examples/212999_data_A.jpg" width="270"/>
.<img src="https://github.com/zampie/Animation-standing-segmentation/blob/master/examples/212999_data_B.jpg" width="270"/>
.<img src="https://github.com/zampie/Animation-standing-segmentation/blob/master/examples/212999_fake_B.jpg" width="270"/>

Usage:

Train:  
Run train.py, edit the file to change parameters.

Get the pretrained model:  
https://drive.google.com/open?id=1SxbJZmwrg8FBR8gScwYGJBxoSlweCRUK

About the dataset:  
.png illustration(Standing) from galgames with alpha channel.  
I can't share them because of the copyright.  
In stead of that, you can download the kancolle standings:  
https://drive.google.com/open?id=1elHLWSpr-T2aSpAttYYMbDnj7sj5vs-Q




Some codes are from: https://github.com/milesial/Pytorch-UNet
