# SuperResolution GAN



This preject implemented SR-GAN algorithm on super resolution.
Paper link: https://arxiv.org/pdf/1609.04802.pdf

The result like this:

![avatar](./images/compare.png)



It's very impressive and do every vivid detail super resolution on low quality images. For now, the pretrained weights is not supported.


## Install

To run this repo, you ganna need install some dependencies.

```
tensorflow-2.0
alfred-py
numpy
matplotlib
opencv-python
```



## Training

We support training on DIV2K dataset, raw data can be obtained from: https://data.vision.ee.ethz.ch/cvl/DIV2K/

After download train/validate images (both HR and LR), soft link them to `./div2k`:

Then, using:

```
python3 train_srgan.py
```

## Models
![avatar](./images/gan.jpeg)
