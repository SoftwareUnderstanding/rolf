# OctaveConvolution
A Gluon Implementation for Drop an Octave

## Usage
- Oct_Resnet V1/V2 is supported.
- Oct_ResNext is supported.

## Result

|Model         |alpha|epochs|batch size|dtype  |tricks         |Top1  |Top5  |param|
|:------------:|:---:|:----:|:--------:|:-----:|:-------------:|:----:|:----:|:---:|
|oct_resnet50v2|0.125|100   |128       |float16|cosine decay   |77.82%|94.13%|[GoogleDrive](https://drive.google.com/open?id=1VAvoqg2brpfELbL1RgLaip6w1NUJAK2W)|

- I use 4 * GTX1080 with pre batch of 128.Cost about 10 days should be better with 2080(ti). 
- I probably should train 120 epochs, break 0.3% acc than proposed.



## Todo
- [ ] mobilenet V2 implement.
- [ ] SE_layer support.
- [ ] ResidualAttention support.

## Paper Reference

[Drop an Octave: Reducing Spatial Redundancy in 
Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)

![](img/OctConv.png)


## Acknowledgment
```
@article{chen2019drop,
  title={Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution},
  author={Chen, Yunpeng and Fan, Haoqi and Xu, Bing and Yan, Zhicheng and Kalantidis, Yannis and Rohrbach, Marcus and Yan, Shuicheng and Feng, Jiashi},
  journal={arXiv preprint arXiv:1904.05049},
  year={2019}
}
```

The [train_script](train_script.py) refers to [Gluon-cv](https://github.com/dmlc/gluon-cv).

```
@article{he2018bag,
  title={Bag of Tricks for Image Classification with Convolutional Neural Networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  journal={arXiv preprint arXiv:1812.01187},
  year={2018}
}
```