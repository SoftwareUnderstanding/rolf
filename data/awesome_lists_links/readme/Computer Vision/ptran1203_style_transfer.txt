
### keras implemetation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf)

![artchitecture](/images/architecture.jpg)

The model architecture proposed by Huang et al., a fixed VGG19 is used to encode both content and style image. The outputs are passed through the adaptive instance normalization (AdaIN) which normalizes the content feature then scale and shift by mean and variance calculated from style feature to have similar global context with the style image. Then, a decoder is used to generate new image from the normalized feature.

------

#### :art: Stylizing result

|  | ![c1](/images/content/brad.png) | ![c1](/images/content/chau.png) | ![c1](/images/content/lance.png) |
|--|--|--|--|
|![c1](/images/style/style_1.png)|![g1](/images/generated/brad_1.png)| ![g1](/images/generated/chau_1.png) | ![g1](/images/generated/lance_1.png) |
|![c1](/images/style/style_2.png)|![g1](/images/generated/brad_2.png)| ![g1](/images/generated/chau_2.png) | ![g1](/images/generated/lance_2.png) |
|![c1](/images/style/style_3.png)|![g1](/images/generated/brad_3.png)| ![g1](/images/generated/chau_3.png) | ![g1](/images/generated/lance_3.png) |
|![c1](/images/style/style_4.png)|![g1](/images/generated/brad_4.png)| ![g1](/images/generated/chau_4.png) | ![g1](/images/generated/lance_4.png) |


#### Setup
- Runtime: Google colaboratory
- Tensorflow `1.15` and Keras `2.2.5`
- Dataset: [Google Scraped Image](https://www.kaggle.com/duttadebadri/image-classification) and [Best Artworks of all times](https://www.kaggle.com/ikarus777/best-artworks-of-all-time)
