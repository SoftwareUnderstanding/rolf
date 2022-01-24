# Project Introduction
This is a project about image style transfer developed by Tao Liang, Tianrui Yu, Ke Han and Yifan Ruan. Our project contains three different models, one is in "cycle_gan_unet" directory which uses the u-net like cnn as generators, one is in "Ukiyoe_codes" directory which uses Resnet blocks as generators, which uses the model proposed in this paper https://arxiv.org/pdf/1703.10593.pdf, the other is in neural_style_transfer that implement sytle transfer using convolution neural network proposed in this paper https://arxiv.org/pdf/1508.06576.pdf.

## Cycle-Gan-Unet
### Description:
This model consists of 2 generators and 2 discriminators. The two generators as U-net like CNNs. During the evaluation of the model, I directly used the pretrained salient objective detection model from Joker, https://github.com/Joker316701882/Salient-Object-Detection.
### Requirements:
Download the check-points for the model from the google drive link, and put them into the corresponding directorys.<br/>
/baroque/checkpoint.pth.tar: https://drive.google.com/open?id=1oMTewhni1L7ZW0F9nNgNoE2RfkrGZ500<br/>
/ukiyo_e/checkpoint.pth.tar: https://drive.google.com/open?id=1mEQliUwOKgSLSUuB_vBXwl03HH_p4VJO<br/>
/salience_model/model.ckpt-200.data-00000-of-00001: https://drive.google.com/open?id=1u8gW2Oj8lZ_Cxqg561lQR9ioDaK64LwX<br/>

### Structure:
./cycle_gan_unet/baroque                         -- Store the checkpoints for baroque style translation<br/>
./cycle_gan_unet/ukiyo_e                             -- Store the checkpoints for ukiyo_e style translation<br/>
./cycle_gan_unet/meta_grapsh                         -- Store the information of the salient objective detection model<br/>
./cycle_gan_unet/salience_model                      -- Store the checkpoints for salient objective detection model<br/>
./cycle_gan_unet/images\*.pkl                        -- All the pickle files are used to store the images according to different styles and landscape<br/>
./cycle_gan_unet/demo.ipynb                           -- This notebook is used for demo, you can choose the image youo want by changing the index of "val_set"<br/>
./cycle_gan_unet/cycle_gan_unet.ipynb                       -- This notebook is the main function of the model<br/>
./cycle_gan_unet/nntools.py                           -- This .py file abstract the status manager and realize the training process of the model<br/>
./cycle_gan_unet/util.py                              -- This .py file is used to realize the image pool called by nntools.py<br/>
./cycle_gan_unet/inference.py                         -- This .py file is used to run the pretrained salient objective detection model<br/>

### Usage:
Directly run the demo.ipynb notebook. You can see the original image and the transferred image.<br/>
If you want to train the model by yourself, delete /baroque and /ukiyo_e directorys. And run the cycle_gan_model.ipynb notebook. You can set all the parameters in the initialization of the experiment class.

## Cycle-Gan-Resnet 
This is the README for photo-to-ukiyoe cycle-GAN style transfer task. Over half of the codes are adopted from 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix' and then modified. The rest are written by student. 

### Requirements:
Install visdom and dominate if willing to display training progress on a webpage by:
    pip install -visdom
    pip install -dominate

### Structure:
single_test.ipynb:   run this notebook to show the Ukiyoe-style transfer result of 'test_image.jpg'. Make sure the image, latest_ukiyoe_G_A.pkl and './models' are in their original places<br/>
train.ipynb:  run this notebook to train a cycle-GAN that can transfer 'datasets/trainA' style to 'datasets/trainB' style. Training options can be found and revised in './options/train_options.py' and './options/base_options.py'<br/>
test.ipynb:  run this notebook to test the model in './checkpoints' file. Input the model name in './options/base_options.py'<br/>
plot_losses.ipynb:   run this to plot losses given a loss log in './checkpoints'<br/>

.Ukiyoe_codes/options/base_options.py:   stores basic training and testing options.<br/>
.Ukiyoe_codes/options/train_options.py:   stores other training options<br/>
.Ukiyoe_codes/options/test_options.py:   stores other testing options<br/>

.Ukiyoe_codes/models/base_model.py:   base class of all the models<br/>
.Ukiyoe_codes/models/cycle_gan_model.py:   implement cycle-GAN model<br/>
.Ukiyoe_codes/models/networks.py:   define basic network behavior methods<br/>
.Ukiyoe_codes/models/test_model.py:   define some testing settings and run the testing from test.ipynb<br/>

.Ukiyoe_codes/util/:   include python files that handle data loading and processing, webpage display and image buffer.<br/>

.Ukiyoe_codes/datasets/:   a folder that stores training and testing data in trainA, trainB, testA and testB subfolders.<br/>

.Ukiyoe_codes/checkpoints/:   a folder storing saved models, loss logs and training options<br/>

.Ukiyoe_codes/latest_ukiyoe_G_A.pkl: the saved generator that can translate images into ukiyoe-style, used in single_test.ipynb<br/>

.Ukiyoe_codes/test_image.jpg: test image used in single_test.ipynb<br/>

### Usage:
single_test.ipynb(for demo use):   run this notebook to show the Ukiyoe-style transfer result of 'test_image.jpg'. Make sure the image, latest_ukiyoe_G_A.pkl and './models' are in their original places<br/>

train.ipynb:  run this notebook to train a cycle-GAN that can transfer 'datasets/trainA' style to 'datasets/trainB' style. Training options can be found and revised in './options/train_options.py' and './options/base_options.py'<br/>

test.ipynb:  run this notebook to test the model in './checkpoints' file. Input the model name in './options/base_options.py'<br/>
plot_losses.ipynb:   run this to plot losses given a loss log in './checkpoints'<br/>



## Neural Style Transfer: 
### Requirements: 
Install package 'pillow' as: $ pip install --user pillow <br/>
Install package 'matplotlib' as: $ pip install --user matplotlib

### Structure:
./neural_style_transfer/Neural_Style_Transfer.ipynb      -- This notebook stores neural style transfer method as well as the demo of the model<br/>
./neural_style_transfer/images                          -- Store the style image and content image for this part, make sure they are in the correct path

### Usage:
Run the Neural_Style_Transfer.ipynb for demo.<br/>
The notebook also stores model. If you want to change the network structure, choose one of content_layers_default and style_layers_default each and comment the others. For white noise input, consider decreasing the weight of style loss and increase the number of optimizing steps. 


