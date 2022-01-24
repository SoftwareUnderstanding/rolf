# DCGAN 
DCGAN implemetation for custom dataset. Inspired from Udacity deep learning nano degree programme.

## Train for your own dataset
1. Keep you images in "data" folder
2. Run train.py script to start training the dcgan
 ```
  python train.py 
 ```
 Model will resize all the images to 32*32. Checkpoints will be stored in "checkpoints" folder.
 
 ## Test the model
 1. Run test.py
  ```
  python test.py 
 ```
 This will take load the checkpoint from checkpoints folder and generate random images.
 
![alt text](https://github.com/Newmu/dcgan_code/blob/master/images/lsun_bedrooms_five_epoch_samples.png)
 
 References:
 Full paper here: http://arxiv.org/abs/1511.06434
 
