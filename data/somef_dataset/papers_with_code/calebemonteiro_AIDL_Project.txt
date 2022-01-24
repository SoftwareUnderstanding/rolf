## SRGAN
Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_ using Keras (tf) for my postgraduate project in Universitat Politècnica de Catalunya.

<p align="center">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/architecture.jpg" width="640"\>
</p>

Paper: https://arxiv.org/abs/1609.04802

## Metrics:

The network is implemented as the paper suggests using perceptual loss as metric to measure the performance of the network.
<p align="center">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/percep_loss.JPG" width="350"\>
</p>

To Extract the the content loss, the paper suggests to use the VGG-19 to calculate the pixel-loss MSE between the features of the Hi-Res image 
and fake Hi-Res image, as it follows:

<p align="center">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/content_loss.JPG" width="350"\>
</p>

## Dataset:
    For SRGAN training we opted to used the CelebA dataset with can be found here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


## Requirements:
    You will need the following to run the above:
    Keras==2.3.1
    tensorflow==2.1.0
    opencv-python==4.3.0
	matplotlib==3.3.0
	argparse==1.4.0
	numpy==1.19.1

## File Structure:
    Model.py   : Contains Generator and Discriminator Network
    Utils.py   : Contains utilities to process images
    train.py   : Used for training the model

## Usage:
    
    Note : During the training the images generated and model will be saved into the directories "images" 
	and "model" following the "sample_interval" parameter. all output folders are automatically created.
    
     * Training (due to my hardware specs, im training with default settings):
        Run below command to start the training process. this script will also download the dataset and prepare the folders needed.
        > python train.py --train_folder='./data/train/' --batch_size=12 --epochs=500 --sample_interval=25


	

## Output:
Below are few results (from epoch 0 to 500):

#### Epoch 0
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_0.png" width="640"\>
</p>

#### Epoch 100
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_100.png" width="640"\>
</p>

#### Epoch 200
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_200.png" width="640"\>
</p>

#### Epoch 300
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_300.png" width="640"\>
</p>

#### Epoch 400
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_400.png" width="640"\>
</p>

#### Epoch 500
<p align="left">
    <img src="https://github.com/calebemonteiro/AIDL_Project/blob/master/resources/epoch_500.png" width="640"\>
</p>

## Findings / Conclusions:
* The architecture suggested by the paper, even with very limited computational resources, is able to archieve some very good results.
* We observed that the network really improves the image quality while applying the upscaling factor by 4 (as suggested by the paper)
* Even when the images are "glitched" the network tries to improve the pixel area of the glitch.
* For some reason, the generator performs badly when dealing with glasses (suggests of samples in the training set/uneven train set or not enough training time)
* If the eyes are nearly closed or too dark (Womans makeup, for example) the generator does not perform very well.
