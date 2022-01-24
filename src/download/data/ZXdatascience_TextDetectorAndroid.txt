# TextDetectorAndroid
This is a Android demo of the text detection based EAST((https://arxiv.org/abs/1704.03155v2)) and MobileNetV2(https://arxiv.org/abs/1801.04381)

The pretrained static model is located at app/src/main/assets/rounded_expanded_convs.pb

The model is small in size because of the modification of EAST using residual bottleneck layer from MobileNetV2.

My modified model architecture can be found:
https://github.com/ZXdatascience/EAST

Here are some examples:

<p float="left">
  <img src="test1.jpg" width="200">
  <img src="test3.jpg" width="200">
  <img src="test4.jpg" width="200">
</p>

