# ML Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements various machine learning models with Python/Tensorflow. I treat mainly "Image processing" in it.

|  Model  |              Paper               |       Status       |
| :-----: | :------------------------------: | :----------------: |
|  U-Net  | https://arxiv.org/abs/1505.04597 | :white_check_mark: |
|  ACoL   | https://arxiv.org/abs/1804.06962 | :white_check_mark: |
| Arcface | https://arxiv.org/abs/1801.07698 | :white_check_mark: |

You can use these models for training or validation.

## Requirements

- Python 3.6>=
- Tensorflow 2.4.0>=
- PIL
- Imgaug
- Numpy
- Scipy
- Matplotlib

I am managing these libraries with pipenv. If you do not have pipenv, install with pip
```bash
pip install pipenv
```
You can see [latest document](https://docs.pipenv.org/en/latest/) to understand the usage more

To install all libraries, you run
```
$ pipenv install
```

## Usage

How to use each model is written in README in the each model. Basically you can training with

```
$ pipenv run python -m $(MODEL_NAME)/train $(options)
```

## Future Plans
- Modularize this repository to enable users to import whole models

## Licence

"ML models" is licenced under the MIT licence.

(C) Copyright 2021, Yudai Hayashi
