# VAE_ABIDE1
This work has been made and shared by the team Institut Hypercube (https://www.institut-hypercube.org/en/, https://www.facebook.com/InstitutHyperCube/).

The model stored here corresponds to the study "Identification of Autism Spectrum Disorders on Brain Structural MRI with Variational Autoencoders" presented at Women in Machine Learning 2018 workshop (files poster4.pdf and Identification_of_Autism_Spectrum_Disorders_on_Brain_Structural_MRI_with_Variational_Autoencoders.pdf)

![alt text](https://raw.githubusercontent.com/blackPacha/VAE_ABIDE1/master/poster43.jpeg)

The architecture of the model in this git repo is: 
- Encoder:
  - 3 (convolutional + max pooling) blocks
  - 2 fully connected layers [2048, 1024]
- bottleneck: 512
- Decoder:
  - 2 fully connected layers [1024, 2048]
  - 3 (deconvolutional + upsampling means) blocks

The activation function used is 'selu': https://www.tensorflow.org/api_docs/python/tf/nn/selu, https://arxiv.org/abs/1706.02515

To use the model: 

- Install Niftynet (https://niftynet.readthedocs.io/en/dev/installation.html)

- Clone blackPacha/vae_abide1/* in your niftynet/extensions; my_collection_network is a python module repository.
(more info: https://niftynet.readthedocs.io/en/dev/extending_net.html)

- Train from niftynet/extensions with command line: 

net_autoencoder train -c my_collection_network/vae_config.ini --name my_collection_network.my_vae.VAE

- Encode and Analyze from niftynet/extensions with command line:

./encode_and_analyze.sh model_number target.npy outdirname

with: 
  - model_number: the number of the model to use;
  - target.npy: a numpy array (n,) being the labels of the n encoded images;
  - outdirname: the absolute path of the directory you want your output goes to
  
- This will produce in outdirname:
  - a directory with all the encoded files
  - an X.npy file which corresponds to the array of all the encoded features ( dim(X) = (number of images, number of latent features) )
  - ROC AUC scores between X and the target before/after selection of features 
