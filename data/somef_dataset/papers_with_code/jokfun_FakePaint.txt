# Fake Paint

Create fake paintings from a data set and a DCGAN model

![Example](output.PNG)

## How to use it

1. Create the dataset. So you can use your own images and add them in a folder, then run :
```
python data.py -p [path_of_the_dataset] -s [size_of_the_img]
```
2. Then, you can train the model. Generated painting will be add in the out folder.
```
python runme.py -s [size_of_the_image] -e [epochs] -b [batch_size] -i [save_interval]
```
__TIPS__ : Depending on the power of your GPU, you may need to adapt the "size" of the training. To do so, you will have to adapt the image size or the batch_size. Moreover, depending on the quality of the data passed in parameter, it is possible that the network is not adapted to your needs, so you will have to review its structure.

## Result

The purpose of this small project is only to test GAN networks. The results are not necessarily very good but I am limited by the hardware I have in my possession.
An update will come if I can apply it on better GPUs.

## Author

Raphael Teitgen [raphael.teitgen@gmail.com]

## Sources : 

The version of this DCGan comes from here : https://arxiv.org/abs/1511.06434
Original code : https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py