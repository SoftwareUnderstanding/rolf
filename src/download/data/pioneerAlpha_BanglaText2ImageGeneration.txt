# Bangla Text to Image  Generation
Pytorch implementaion of Attentional Generative Adversial Network ([AttnGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf)) for Bengali Language

<p align="center">
  <img src="examples/archi.png" align="center" width="1000" height="400" />
</p>

<!-- <p align="center"><b><i>Sample generated results of our model. First row contains images of three different resolutions (low-to-high). Second and third row represents top-5 most attented words of out attention model.</i></b></p>
<p align="center">
  <img src="examples/t2i_home.png" align="center" width="1000" height="600" />
</p> -->

<p align="center"><b><i>Attention maps of text-to-image synthesis by our Attentional GAN model.</i></b></p>
<p align="center">
  <img src="examples/attntion.png" align="center" width="700" height="400" />
</p>


<p align="center"><b><i>examples of text-to-image synthesis by our Attentional GAN model.</i></b></p>
<p align="center">
  <img src="examples/T2I_samples.png" align="center" width="1000" height="400" />
</p>

<p align="center"><b><i>Sample generated bengali text to image results</i></b></p>

| "রঙিন পাখি যার একটি নীল মাথা মুখ এবং  পেট সাদা এবং সাদা নীল পালক" | "ছোট পাখির একটি ছোট বিল কালো উইংবার সাদা বুক এবং একটি সোনার গলা রয়েছে" <img width=240/>| "একটি ছোট পাখি যার উজ্জ্বল লাল মুকুট এবং পেট বাদামী ডানা এবং বাদামী গালের প্যাচ" |
|:--:|:--:|:--:|
<img src="examples/blue.png" width="240" height="240"/> | <img src="examples/black.png" width="240" height="240"/> | <img src="examples/red.png" width="240" height="240"/> |



### Dependencies
python 3.6+

Pytorch 1.0+

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`

### Copied from LICENSE file (MIT License) for visibility:
*Copyright for most part of this project bengali T2I are held by Tao Xu, 2018 as part of project AttnGAN. All other copyright for project Bengali T2I are held by this project owner, 2021. __All non-data files that have not been modified by owner include the copyright notice "Copyright (c) 2018 Tao Xu" at the top of the file.__*


**Reference**

- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/taoxugit/AttnGAN)
- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) [[code]](https://github.com/carpedm20/DCGAN-tensorflow)


### TODO:
- [ ] preprocessed meta data and dataset 
- [ ] Training 
- [ ] pretrained model
- [ ] valiadation
- [ ] Deploy as a web app that makes it easy to control the specific image one wants to generate
