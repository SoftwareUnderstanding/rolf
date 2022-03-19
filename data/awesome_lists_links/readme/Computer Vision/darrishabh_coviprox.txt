# **COVIPROX: A DEEP LEARNING FRAMEWORK FOR SURVEILLANCE BASED CONTACT TRACING** 

![Image](/Images/model_flow.png)

### **What is the Aim of this Work?**
This work aims to estimate the probability of a person contracting COVID from another person present in a CCTV footage based on 2 parameters: Whether the two people of interest are wearing a mask and the distance between the two people. 
For this we utilise a mask detector model ([vgg_19 with batch normalisation](https://arxiv.org/abs/1409.1556)) and [TransREID](https://arxiv.org/pdf/2102.04378.pdf) for Person detection across the frames of the CCTV footage. A distance metric and an empirical formula for probability calculation is utilised.


### **Installing the necessary packages**
This work is done in python. you will require pip for installation. Create a virtual environment and in the virtual environment install the dependencies from the requirements.txt. 

```
pip install -r requirements.txt
``` 
### **Running the Program**
The program can be run from the command line using the following syntax

```
python main.py --config_path {1} --img_path {2} --mask_path {3} --gal_path {4}
```
**where** 

**1**:  Represents the path for the config file. The config file is the transformer_base.yml located in the configs folder in REID.

**2**: Represents the Path towards the test image. 

**3**: Represents the path towards the mask detector model

**4**: Represents the path of the database where the comparison images    are stored. Commonly known as gallery images in person reidentification

Also in **transformer_base.yml** in the **configs** folder in **REID** change the **PRETRAIN_PATH** to your own pretrained model path for transreid. 

## References for Models 
- He, S., Luo, H., Wang, P., Wang, F., Li, H., & Jiang, W. (2021). Transreid: Transformer-based object re-identification. arXiv preprint arXiv:2102.04378.
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

