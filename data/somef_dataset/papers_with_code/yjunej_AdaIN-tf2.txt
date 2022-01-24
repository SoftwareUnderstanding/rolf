# AdaIN-tf2
Simple AdaIN implements using TF2
Based on: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" https://arxiv.org/abs/1703.06868

Mini AdaIN layer Only for one pair of images

## Architecture

![image](https://user-images.githubusercontent.com/61140071/128589787-5ef50472-0c4d-4c56-9535-9a3048149c1b.png)

## Experiment

- Adam Optimizer (learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
- MSE
- half size random cropping
- pretrained VGG19

## Results

|Content|Style|Stylized Image
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/yjunej/AdaIN-tf2/raw/master/data/content.jpg">|<img src="https://github.com/yjunej/AdaIN-tf2/raw/master/data/style.jpg">|<img src="https://github.com/yjunej/AdaIN-tf2/raw/master/result/result.gif">


