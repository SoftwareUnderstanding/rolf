<h1 align="center">Welcome to GAN-Pytorch 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <img src="https://img.shields.io/badge/python-%3E%3D3.7-orange.svg" />
  <img src="https://img.shields.io/badge/pytorch-%3E%3D1.6.0-orange.svg" />
  <img src="https://img.shields.io/badge/torchvision-%3E%3D0.7.0-orange.svg" />
  <a href="https://ainote.tistory.com/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>

</p>

> Generating Images Using GAN (Dogs, Cats..) w/ Pytorch

### 🏠 [Homepage](https://github.com/myoons/GAN-pytorch/blob/master/README.md)

### ✨ [Description](https://ainote.tistory.com/5)

## [Data](https://www.kaggle.com/prasunroy/natural-images)

Store images as data/{--object}/imageSequence/00001.png  
object is an argument that the GAN Network will generate.  
List of Objects = [airplane, car, cat , dog, flower, fruit, motorbike, person] / Default : dog

## Prerequisites

- python >= 3.7
- pip >= 20.1.1
- pytorch >= 1.6.0
- torchvision >= 0.7.0

## Install

```sh
install -r requirements.txt
```

## Usage

```sh
python main.py
```

## Author

👤 **Yoonseo Kim**

* Website: https://ainote.tistory.com/
* Github: [@myoons](https://github.com/myoons)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/myoons/GAN-pytorch/issues). 

## Show your support

Give a ⭐️ if this project helped you!

## References
1. Dataset (https://www.kaggle.com/prasunroy/natural-images)
2. Paper (https://arxiv.org/pdf/1406.2661.pdf)
3. Basecode (https://github.com/eriklindernoren/PyTorch-GAN)