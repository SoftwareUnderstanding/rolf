Generative Adversarial Network(GAN)
==============================
Implemented DCGAN which is a direct extension of the GAN.

## Table of Contents
* [Getting Started](#getting-started)
* [Install](#install)
* [Execute The Project](#how-to-run-this-project)
* [Features Released](#features-released)
* [Upcoming Features](#upcoming-features)
* [Maintainers](#maintainers)
* [Contributes](#contributes)
* [Disclaimer](#disclaimer)
* [References](#references)
* [License](#license)

## Getting Started
These instructions will get you a copy of the project up and running on your development machine for contributing to development, testing purposes.

### Network Architecture of DCGAN
![Architecture of DCGAN](dcgan_generator.png)

#### Python Version
> Minimum python version should have 3.x.x or upper


## Install
A step by step series of examples that tell you have to get a development env running

### How do I get set up?
If you would like to used `Virtualenv`
Install the virtualenv using this command(If you have not installed virtualenv yet.)

```python
$ [sudo] pip install virtualenv
```
Learn more to visit [Virtualenv](https://virtualenv.pypa.io), [User Guide](https://virtualenv.pypa.io/en/stable/userguide/)

Create a directory using the following command from your terminal
```ssh
$ [sudo] mkdir GAN
```

Switch to **GAN** directory
```ssh
$ cd GAN
```

#### git clone
```python
$ git clone git@github.com:porimol/GAN.git .
```

Afterthen, create virtual env file by the following command from your terminal
```ssh
$ virtualenv -p python3 .venv
```

If you create virtual env file successfully on your development machine then run this command
```ssh
$ source .venv/bin/activate
```

Installing the requirements using the following commands
```python
$ pip install -r requirements.txt
```

## How to run this project
```python
$ cd GAN
$ python dcgan.py
```

## Features Released

* [x] [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks(DCGAN)](https://arxiv.org/pdf/1511.06434.pdf)

## Upcoming Features

* [ ] Deploy demo using Flask

## Maintainer
***Porimol Chandro*** \
[Linkedin Profile](https://www.linkedin.com/in/porimolchandro/)

## Contributors
* ****

## Contributing
See the list of [contributors](https://github.com/porimol/GAN/contributors) who participated in this project.


### How to become a contributor

If you want to contribute to `GAN` and make it better, your help is very welcome.
You can make constructive, helpful bug reports, feature requests and the noblest of all contributions.
If like to contribute in a good way, then follow the following guidelines.

#### How to make a clean pull request

* Create a personal fork on Github.
* Clone the fork on your local machine.(Your remote repo on Github is called `origin`
* Add the original repository as a remote called `upstream`.
* If you created your fork a while ago be sure to pull upstream changes into your local repository.
* Create a new branch to work on! Branch from `dev`.
* Implement/fix your feature, comment your code.
* Follow `GAN`'s code style, including indentation(4 spaces).
* Write or adapt tests as needed.
* Add or change the documentation as needed.
* Push your branch to your fork on Github, the remote `origin`.
* From your fork open a pull request to the `dev` branch.
* Once the pull request is approved and merged, please pull the changes from `upstream` to your local repo and delete your extra branch(es).


## Disclaimer

This repository is not ready as production grade and it is being implemented in the contributor's free time, and as such, may contain minor errors in regards to some portion of the code.

### Inspired By
> PyTorch DCGAN Tutorial


## References
 [1] Radford, A., Metz, L. and Chintala, S., 2015. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.\
 [2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).



## License
### [The MIT License](LICENSE)

Copyright (c) 2020, Porimol Chandro <porimolchandroroy@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.