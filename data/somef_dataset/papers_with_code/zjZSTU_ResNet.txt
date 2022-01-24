# ResNet

[![Documentation Status](https://readthedocs.org/projects/resnet/badge/?version=latest)](https://resnet.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> `ResNet`算法实现

实现`ResNet`及后续版本，同时实现了相关模型：

* `ResNet v1/v2`
* `DenseNet`

| CNN Architecture | Data Type (bit) | Model Size (MB) | GFlops （1080Ti） | Top-1 Acc(VOC 07+12) | Top-5 Acc(VOC 07+12) |
|:----------------:|:---------------:|:---------------:|:-----------------:|:--------------------:|:--------------------:|
|     ResNet-18    |        32       |      44.607     |       3.641       |        89.98%        |        99.29%        |
|     ResNet-34    |        32       |      83.180     |       7.348       |        90.01%        |        99.29%        |
|   ResNet-34_v2   |        32       |      83.177     |       7.349       |        90.50%        |        99.29%        |
|     ResNet-50    |        32       |      97.492     |       8.223       |        89.37%        |        99.39%        |
|    ResNet-101    |        32       |     169.942     |       15.668      |        90.66%        |        99.35%        |
|   ResNet-101_v2  |        32       |     169.926     |       15.668      |        90.85%        |        99.48%        |
|   DenseNet-121   |        32       |      30.437     |       5.731       |        89.86%        |        99.20%        |

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
* [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
* [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## 安装

### 文档工具依赖

```
# 文档工具依赖
$ pip install -r requirements.txt
```

### python库依赖

```
$ cd py
$ pip install -r requirements.txt
```

## 用法

### 文档浏览

有两种使用方式

1. 在线浏览文档：[ResNet](https://resnet.readthedocs.io/zh_CN/latest/)

2. 本地浏览文档，实现如下：

    ```
    $ git clone https://github.com/zjZSTU/ResNet.git
    $ cd ResNet
    $ mkdocs serve
    ```
    启动本地服务器后即可登录浏览器`localhost:8000`

## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 致谢

### 引用

```
@misc{he2015deep,
    title={Deep Residual Learning for Image Recognition},
    author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    year={2015},
    eprint={1512.03385},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{he2016identity,
    title={Identity Mappings in Deep Residual Networks},
    author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    year={2016},
    eprint={1603.05027},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{huang2016densely,
    title={Densely Connected Convolutional Networks},
    author={Gao Huang and Zhuang Liu and Laurens van der Maaten and Kilian Q. Weinberger},
    year={2016},
    eprint={1608.06993},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}

@misc{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}
```

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/ResNet/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjZSTU
