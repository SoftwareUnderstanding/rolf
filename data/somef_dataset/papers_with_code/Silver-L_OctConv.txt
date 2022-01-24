# Ocatve Convolution (OctConv)
This is a experiment of OctConv with ResNet-50 on CIFAR-10/CIFAR-100

Y. Chen, H. Fang, B. Xu, Z. Yan, Y. Kalantidis, M. Rohrbach, S. Yan, J. Feng. Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution. (2019). https://arxiv.org/abs/1904.05049

## Requirment
```
1. Tensorflow==1.13.1
2. tqdm
```

## Architecture of OctConv
<img src="https://github.com/Silver-L/OctConv/blob/master/result/octconv.png" width="660" height="400" alt="error"/>

## Training
Training network on CIFAR-10/CIFAR-100
```
python train.py --problem cifar10/cifar100
```

Training the normal ResNet-50
```
python train.py --is_octconv=false
```

Training the OctConv ResNet-50
```
python train.py --is_octconv=true
```

## Result
### Dataset: CIFAR-10

Training loss \
<img src="https://github.com/Silver-L/OctConv/blob/master/result/train_loss.png" width="600" height="300" alt="error"/>

Test accuracy \
<img src="https://github.com/Silver-L/OctConv/blob/master/result/test_accuracy.png" width="600" height="300" alt="error"/>

|               | Parameters       |
| --------------|:----------------:|
| Normal Resnet |38,103,690(100%)  |
| OctConv-Resnet|24,628,068(64.5%)  |

|               | alpha | Test accuracy |
| --------------|:-----:|:-------------:|
| Normal Resnet |0      |     84.11%    |
| OctConv-Resnet|0.125  |  ***85.36%*** |
| OctConv-Resnet|0.25   |     84.83%    |
| OctConv-Resnet|0.5    |     82.48%    |

## Reference
```
1. https://github.com/koshian2/OctConv-TFKeras
2. https://github.com/taki0112/ResNet-Tensorflow
```
