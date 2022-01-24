# MobileNet_Zoo
A Keras implementation of MobileNet_V1 and MobileNet_V2.

## Requirement
- OpenCV 3
- Python 3
- Tensorflow or Tensorflow-gpu
- Keras

## Training with Keras

 - Training the classification model of cifar-10.
```
python3 train.py [--net]
# --net in {mobilenet_v1, mobilenet_v2}
```

## Visualizating with Tensorboard

 - Visualizating the classification model of cifar-10.
```
tensorboard --logdir=logs/[net]/000
# --net in {mobilenet_v1, mobilenet_v2}
```

## Performance

This is the timing of MobileNetV1 vs MobileNetV2 using TF-Lite on the large core of Pixel 1 phone.
<div align="center">
<img src="https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mnet_v1_vs_v2_pixel1_latency.png"><br><br>
</div>

## Reference

- [MobileNet_V1](https://arxiv.org/abs/1704.04861)
- [MobileNet_V2](https://arxiv.org/abs/1801.04381)
