# fixmatch-fastai

This is an implementation of FixMatch in fastai.

This semi-supervised learning algorithm combines consistency regularization and pseudo-labelling to make use of unlabeled data.
Weakly-augmented unlabeled images are fed to a model.
If a prediction is above a confidence threshold, it is retained as a pseudo-label. Then, the model is trained to predict the same pseudo-label from a strongly-augmented version of the same image.

https://arxiv.org/abs/2001.07685

<img src="https://miro.medium.com/max/1400/1*5SCSOqvXcrxL-IwZmZaH_g.png"/>

This still needs to be improved.
But it can run with any custom dataset and can take any fastai or torchvision transforms

Inspiration for this code:

https://github.com/oguiza/fastai_extensions/blob/master/04a_MixMatch_extended.ipynb

https://github.com/kekmodel/FixMatch-pytorch
