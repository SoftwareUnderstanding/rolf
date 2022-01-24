# image-superres
Implementation of image super-resolution using generative adversarial network (GAN)
The paper from which the model has been taken for implementation is available here : https://arxiv.org/pdf/1609.04802.pdf
The model consists of two networks: generator and discriminator. The generator is used to obtain the high resolution image from the low resolution image and discriminator is used to differentiate between original higher resolution image and generator super resolved image. Discriminator is used in training to train the generator to generate super-resolved images closer to the original high resolution image. Simultaneously, discriminator is also trained to better differentiate between true high resolution and generator super-resolved images.
## Generator network
![gen_net](https://user-images.githubusercontent.com/24764839/127691842-b587444e-3cae-4ceb-9d56-7730a47dd502.JPG)
## Discriminator network
![disc_net](https://user-images.githubusercontent.com/24764839/127691899-6ada2363-a44f-4aba-99d6-9d9b6ecf0ff6.JPG)
## build_model
build_model.py provides functions 'build_generator_net' and 'build_discriminator_net' to create generator and discriminator models respectively.
### build_generator_net(B)
Creates the generator model as depicted in the figure above where B is the number of residual blocks.
### build_discriminator_net(n,k,s)
Creates the discriminator model as shown in the figure above. n, k and s respectively signify the number of filters, kernel size and strides respectively.
##loss_calc
Provides functions for calculation of loss.
### vgg_loss(true_hr, false_hr, VGG_MODEL, output_layer='block5_conv4')
Calculates the feature loss by comparing the embeddings of true_hr and false_hr after passing it through VGG_MODEL till the output_layer. 'true_hr' is the true high resolution image, 'false_hr' is the generated super-resolution image, 'VGG_MODEL' is the pretrained downloaded VGG model and 'output_layer' (default 'block5_conv4': 4th convolutional layer in the 5th recursive block of the VGG model) signifies the layer in the VGG model from which the image embeddings are to be calculated.
### adv_loss(disc_output)
Calculates the adversarial loss of the model. The output of the discriminator network when generated super-resolution images are passed to it is used as the input for this function. Total generator loss is a weighted addition of adversarial loss and generator loss.
### disc_loss(true_outputs, false_outputs)
Returns sum of 'true_loss' and 'false_loss'. 'true_loss' is the loss when true high-resolution images are passed to the discriminator net and 'false_loss' is the loss when generated high-resolution images are passed to the discriminator net. 'true_outputs' is the output of the discriminator when true high-res images are passed to it and 'false_outputs' is the output of the dicriminator when generated super-res images are passed to it.
## prepare_data(folderpath)
Preprocesses the images in 'folderpath'. 96x96 crops are taken from the images which are the true high-res images and resized to 24x24 and used as the low-res images to be super-resolved.
## trainer
Used for training the model
### train(generator_model, discriminator_model, VGG_MODEL, dataset, epochs=10)
Generator and discriminator models are passed using 'generator_model' and 'discriminator_model' respectively. 'VGG_MODEL' is the pretrained dowloaded VGG model used for calculating vgg_loss. 'dataset' contains batched training data of low-res and high-res images. 'epochs' (default 10) specifies the number of epochs for which training should be done on the models. This function calls train_step for each batch in each epoch.
### train_step(generator_model, generator_opt, discriminator_model, discriminator_opt, VGG_MODEL, true_lr, true_hr)
Where the actual training happens. 
## for_training.ipynb
Use this notebook for training the model.
## for_testing.ipynb
Use this notebook for testing the model.
