# ws-preprocess
This is image restoration for UAV based wildfire segmentation, because it will always meet some disturbance, noise or other serious situation. 
The code is based on the fastai packages and the [ fastai class 7](https://course.fast.ai/videos/?lesson=7).  
The thought may comes from antique fraud. It is assumed that we want to produce a fake antique, and we got other authentic products during its specific age. After learning their features and finish making the fake, we show it to the expert and the expert tells us where we made it wrong. Then, every time we adjust more features make it harder for that expert to tell which is counterfeit.  


## Building the environment 
* Install the Google compute platform on Linux
```
# Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk
```
  
* Conda install fastai packages
```
sudo /opt/anaconda3/bin/conda install -c fastai fastai
```
**Fastai give us many ideas to avoid running out of GPU RAM, like the `mixed procision training` to make us train the model on 16 bite position.** This is mentioned in [fastai class 3](https://course.fast.ai/videos/?lesson=3).

## [GAN1.ipynb](https://github.com/qiaolinhan/ws-preprocess/blob/master/GAN1%20.ipynb)
This model is the fist part of the image restoration model.    
This is a **generative adversarial netork (GAN) model**.    
  * **Generator**: Resnet34 pretrained U-net.
  * **Discriminator**: A binary classifier with adaptive binary cross entropy loss, which is foun by fastai function `gan_critic()`.  
  * Using spectral normalization, based on [this paper](https://arxiv.org/abs/1802.05957): https://arxiv.org/abs/1802.05957.  
## [GAN2.ipynb](GAN2.ipynb)
This model is the second part of the image restoration model.  
This is a **loss network critic U-net model**.  
  * **U-net**: Resnet34 pretrained.  
  * **Loss network**: Imagenet pre-trained VGG16, grab the feature maps or the activations befor thire changing (one step before max pooling) in middle of this network.  
  * Model based on [the paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43): https://link.springer.com/chapter/10.1007/978-3-319-46475-6_43.


## [Data](https://github.com/qiaolinhan/ws-preprocess/tree/master/Data)
The data we used comes from Google Image and they are cleared and stored in the folder `target`, in the crappify step, these images are crappied into folder `crappy`.  
Other folders are the processing from 'crappy' to 'target'. The network models will be stored in the processing.

-------------------------
If you think its interesting, please star us or cite our paper :)

[Qiao, Linhan, Youmin Zhang, and Yaohong Qu. "Pre-processing for UAV Based Wildfire Detection: A Loss U-net Enhanced GAN for Image Restoration." 2020 2nd International Conference on Industrial Artificial Intelligence (IAI). IEEE, 2020.](https://ieeexplore-ieee-org.lib-ezproxy.concordia.ca/abstract/document/9262172)
```
@inproceedings{qiao2020pre,
  title={Pre-processing for UAV Based Wildfire Detection: A Loss U-net Enhanced GAN for Image Restoration},
  author={Qiao, Linhan and Zhang, Youmin and Qu, Yaohong},
  booktitle={2020 2nd International Conference on Industrial Artificial Intelligence (IAI)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

```
