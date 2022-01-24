### Facial Expression Recognition

---

**Facial expression recognition** is the task of classifying the **expressions** on **face** images into various categories such as anger, fear, surprise, sadness, happiness and so on. 

### KDEF dataset

---

* The **Karolinska Directed Emotional Faces** (**KDEF**) is a set of totally 4900 pictures of human facial expressions.
* You will find more details about [KDEF *here*](https://www.kdef.se/home/aboutKDEF.html).

Official Dataset : [KDEF](https://www.kdef.se/)

![img](https://www.kdef.se/____impro/1/onewebmedia/ContactSheet_001.jpg?etag=W%2F%221ee93-59942755%22&sourceContentType=image%2Fjpeg&quality=85)



### Efficient-Net: Rethinking Model Scaling for Convolutional Neural Networks

---

![img](https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s640/image2.png)

* **EfficientNet** was published in 2019 at the [International Conference on Machine Learning](https://icml.cc/Conferences/2019) (ICML). On the ImageNet challenge, with a **66M** parameter calculation load, EfficientNet reached 84.4% accuracy and took its place among [the state-of-the-art](https://paperswithcode.com/sota/image-classification-on-imagenet).

  ![img](https://miro.medium.com/max/691/1*5oQHqmvS_q9Pq_lZ_Rv51A.png)

  ​													     **Architecture of Efficient-Net**

* **EfficientNet** can be considered a group of convolutional neural network models. But given some of its subtleties, it’s actually more efficient than most of its predecessors.

* The **EfficientNet** model group consists of 8 models from **B0 to B7**, with each subsequent model number referring to variants with more parameters and higher accuracy.

**How It works**

---

![img](https://miro.medium.com/max/873/0*r01mB4rWO1chqBAO)

- **Depthwise Convolution + Pointwise Convolution:** Divides the original convolution into two stages to significantly reduce the cost of calculation, with a minimum loss of accuracy.
- **Inverse Res:** The original ResNet blocks consist of a layer that squeezes the channels, then a layer that extends the channels. In this way, it links skip connections to rich channel layers. In MBConv, however, blocks consist of a layer that first extends channels and then compresses them, so that layers with fewer channels are skip connected.
- **Linear bottleneck:** Uses linear activation in the last layer in each block to prevent loss of information from ReLU.

### Efficient-Net Result in Facial Expression

---

![](https://github.com/ChintanThacker/Facial_Expression_Effiecentnet/blob/master/effnet.jpg)

**References:**

---

1. https://arxiv.org/abs/1905.11946
2. https://heartbeat.fritz.ai/reviewing-efficientnet-increasing-the-accuracy-and-robustness-of-cnns-6aaf411fc81d
