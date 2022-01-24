# GAN Image Editor
#### By Devang Antala, Lee Cowan, Charles Nguyen, Myra Zubair

GAN Image Editor is a responsive web application that utilizes the ability of GAN (Generative Adversarial Networks) to edit and generate images in real time. A user can upload an image (given that it fits certain parameters) and have the ability to alter specific features of that image. Currently, the web app showcases three types of image alteration that use GAN technology: facial editing, clothing style swapping, and image quality enhancement. These features give the user various methods of customization and allows them to reach their goals without the need of any image editing software or expertise. The website also allows users to register an account and be able to store their images on the web server. This way, users can view their uploaded images alongside multiple edited versions of those same images without needing to store them locally or use any image viewing program.

## Features

- Facial Attribute Editing
	- Edit up to 13 pretrained features in a face image. 
		- Including changes hair color, skin tone/color, age of face, and more.
	- Control the intensiy of each feature you choose to edit. 
	- Edit your own custom image using facial GAN. Model can function on any input size of image. 
	- Provide 2 seperate models that can run.
		- Pre-trained model resulting in image of size 384x384
		- Custom model resulting in miage of size 128x128
- Low resolution to high resolution
	- An image of low resolution will be converted to higher resolution
		- Pretrained model has been trained for 1000000 iterations
		- Custom model has been trained for 9880000 iterations
- Style transfer
	- After two images are selected, the clothing on the second picture will transfer and replace the clothing on the first image
	- Contains twe separate models, trained on 256x256 dataset:
		- Pre-trained model has been trained for 30 iterations
		- Custom model has only been trained for 15 iteratons.

### Bugs

- StyleGAN
	- When running StyleGAN multiple times in a row, it may not update but instead show the last generated image. 
	- Every time the user tries to run StyleGAN, the user may need to refresh the webpage before proceeding

## Installation and References

### Web

- Database credentials need to be edited in DbConn.java (in the package DbUtils) in order to use the site's functionalities:
	- dbAndPass, a string that specifies the database, user login name and password.
	- DRIVER, a string that specifies the database driver used.
	- url, a string that specifies the url for the database.
	- isTemple(), a function that determines if the server is running on Temple's network.
	- user_table and image_table, database tables needed for the server requests. More specifics can be found in other documentation.

- Port numbers need to be changed in the following files to match what the Flask server is listening to:
	- testFlask.js
- displayFacialGAN.js
	- displayStyleGAN.js
 	- displayQualityGAN.js

- File paths also need to be changed to correspond to different deployment environments.
- File paths are semi-hard coded in various JSP and js files.

### Facial GAN

- In order to run the facial GAN, the website should be set up initially to see bugs and error message. See web section to determine changes that need to be made there. 

- Download the gan_models folder and store it on the server that you would like to host it on. 

- Dependies for the facial GAN:
	- Python 2.7 or 3.6
	- Tensorflow 1.7
	- Optional: CUDA Toolkit can be installed to improve training time if using nvidia.  
		- https://developer.nvidia.com/cuda-toolkit-archive
	- Full enviromnment of python packages listed in /GAN_Image_Editor/gan_models/facial_editing_gan/packages.txt in this repo
	
- Once dependencies are installed, type the following command in your development environment in order to start to flask server for facial GAN. Default port that the flask server for facial app is run on is 7001.

	```console
	python facial_app.py
	```
	
- The facial GAN allows you to edit the attribute intensity and appearence in a given portrait image. 
	- The image must be in either .png or .jpeg format.
	- The facial GAN currently supports a set of 13 selectable attributes.
		- Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young

- In order to run the facial GAN without the website, you can utilize the following command. 

	```console
	python test_slide1.py --experiement_name [EXPERIMENT NAME] --test_att [TEST ATTRIBUTE] --test_min [TEST MIN] --test_max [TEST MAX] --n_slide [NUM INTENSITY]
	```
	- The above command only requires the experiment name and test attribute as mandatory, the rest will take default parameters. 
	

- The pre-trained models are available below. The models are for 384x384 and 128x128 images. 
	- Download custom model here: https://drive.google.com/open?id=1kVXjjcWSOwwErA1jDBh9U5mqEqlAQnh7 (
	Must be Temple Univeristy student to access)
		- Simply download and unzip folder to GAN_Image_Editor/gan_models/facial_editing_gan/output
		- Run the command shown above and replace EXPERIMENT NAME with the name of the folder in ./output.
			- Default name is 128_custom.
	- Download pre-trained model here:
		- https://github.com/LynnHo/AttGAN-Tensorflow
	
- References
	- https://github.com/LynnHo/AttGAN-Tensorflow
	- https://arxiv.org/abs/1711.1067

### Style GAN

The StyleGAN implementation for the GAN Image Editor is built on Impersonator. Although there are two additional capabilities of Impersonator (Human Motion Imitaiton and Novel View Synthesis), the scope of this project is limited to the Appearance Transfer due to time and resource constraints.

<p float="center">
	<img src='gan_models/style_gan/impersonator/assets/visuals/motion/Sweaters-id_0000088807_4_full.jpg' width="135"/>
  	<img src='gan_models/style_gan/impersonator/assets/visuals/motion/mixamo_0007_Sweaters-id_0000088807_4_full.gif' width="135"/>
  	<img src='gan_models/style_gan/impersonator/assets/visuals/appearance/Sweaters-id_0000337302_4_full.jpg' width="135"/>
	<img src='gan_models/style_gan/impersonator/assets/visuals/appearance/Sweaters-id_0000337302_4_full.gif' width="135"/>
	<img src='gan_models/style_gan/impersonator/assets/visuals/novel/Jackets_Vests-id_0000071603_4_full.jpg' width="135"/>
    <img src='gan_models/style_gan/impersonator/assets/visuals/novel/Jackets_Vests-id_0000071603_4_full.gif' width="135"/>
    <img src='gan_models/style_gan/impersonator/assets/visuals/motion/009_5_1_000.jpg' width="135"/>    
  	<img src='gan_models/style_gan/impersonator/assets/visuals/motion/mixamo_0031_000.gif' width="135"/>
  	<img src='gan_models/style_gan/impersonator/assets/visuals/appearance/001_19_1_000.jpg' width="135"/>
	<img src='gan_models/style_gan/impersonator/assets/visuals/appearance/001_19_1_000.gif' width="135"/>
	<img src='gan_models/style_gan/impersonator/assets/visuals/novel/novel_3.jpg' width="135"/>
    <img src='gan_models/style_gan/impersonator/assets/visuals/novel/novel_3.gif' width="135"/>
</p>

#### Getting Started

- A conda virtual environment export was set up with a majority of the depenedencies in an all-in-one place. Try this first:

```
. /opt/anaconda3/etc/profile.d/conda.sh   

cd impersonator
conda env create
conda activate swapnet
```

- Set up `Impersonator` and make sure it is functioning before proceeding. if impersonator runs into any issues on the GPU machine, go to the `gan_models/style_gan/impersonator/readme.md` for further instructions on the set up.

- Also be sure you are running the virtual environment every time you are using impersonator by running:

```
. /opt/anaconda3/etc/profile.d/conda.sh   
conda activate swapnet
```

- The training dataset can be downloaded from [OneDrive](https://onedrive.live.com/?authkey=%21AJL_NAQMkdXGPlA&id=3705E349C336415F%2188052&cid=3705E349C336415F). This should include `PER_256_video_release.zip`, `smpls.zip`, `train.txt`, and `val.txt`.

- The files should be moved and extracted to `impersonator/data/iPER`.
Please check `GAN_Image_Editor/gan_models/style_gan/impersonator/readme.md` and `GAN_Image_Editor/gan_models/style_gan/impersonator/doc/train.md` for more details on setting up the pretrained models and the dataset.


#### Setting Up Server

- Start up the server by running:
```
. /opt/anaconda3/etc/profile.d/conda.sh   
conda activate swapnet
cd impersonator
python charles_app.py
```

- This allows the webpage to call the Style GAN and let it run impersonator whenever called.

### Image GAN

The Image GAN implementation is based on SRGAN (Super Resolution Generative Neural Network). The website should be functional in order for the SRGAN to send and receive images. Download the files in thr Quality Gan folder into the directory where you will start up your server from.

You can download the pretrained model from https://drive.google.com/a/temple.edu/uc?id=0BxRIhBA0x8lHNDJFVjJEQnZtcmc&export=download.
Download the VGG19 weights from the http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
You can download the High Resolution training dataset from https://drive.google.com/file/d/0BxRIhBA0x8lHYXNNVW5YS0I2eXM/view?usp=sharing
You can download the Low Resolution training dataset from https://drive.google.com/file/d/0BxRIhBA0x8lHNnJFVUR1MjdMWnc/view?usp=sharing

#### Setting Up Server

Start up the server by running:
```
 . /opt/anaconda3/etc/profile.d/conda.sh; conda activate tensorflow_gpuenv   
python myra_app.py

References:
https://github.com/brade31919/SRGAN-tensorflow
https://arxiv.org/pdf/1609.04802.pdf
