# img-colorization
This repository will contain the codes and information from my Image Colorization project using PyTorch. 

The purpose of the notebook is to let myself be familiar with the process of training and testing a DL model. Image colorization was chosen as the topic due to the large amount of resources available online. Much of the codes were taken from an article here: https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8, with slight modifications made to the model itself. A different dataset was used and the way the data was extracted, transformed and loaded varies from the article too. Other resources referenced in the notebook include the papers: https://arxiv.org/abs/1603.08511, and https://arxiv.org/abs/1611.07004. 

More on the dataset used in the notebook, COCO dataset was used as the article did too, but I had combined another 1,000 images from ImageNet making up the total of 10,000 images for training.

The structure of the model consists of a U-net Generator and a Patch Discriminator as the main networks for this GAN. The results on the notebook is just after 1 epoch and for demonstration purposes. Results have been added in another folder.

To end, this is just a practice for Deep Learning model deployment for me to better understand the whole flow and will be a stepping stone for more complicated projects in the future.
