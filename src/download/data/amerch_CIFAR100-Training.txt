# cifar100

Repository Used for Training ResNet 18 Models on CIFAR100 in Google COLAB. Extension of code used from original mixup paper.

This extended previous code to allow for models to be trained using:
1. mixup (from [https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412 "mixup: Beyond Empirical Risk Minimization")).
2. adversarial training using FGSM (from [https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572 "Explaining and Harnessing Adversarial Examples")).
3. adversarial training using PGD (from [https://arxiv.org/pdf/1706.06083](https://arxiv.org/pdf/1706.06083 "Towards Deep Learning Models Resistant to Adversarial Attacks")).
4. Vertical and Horizontal Translations (related to [https://arxiv.org/abs/1805.12177](https://arxiv.org/abs/1805.12177 "Why do deep convolutional networks generalize so poorly to small image transformations?")). 
