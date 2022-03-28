# CAPTURING HUMAN CATEGORY REPRESENTATIONS BY SAMPLING IN DEEP FEATURE SPACES

Paper Link:     https://openreview.net/forum?id=BJy0fcgRZ

Referenced Paper Link:       https://arxiv.org/abs/1605.09782 

----

## Introduction
In the paper writers used 2 GAN types which are DCGAN and BiGAN classify images.
DCGAN wass used for creating happy, sad, male, female faces.
BiGAN, which employed from Adversarial Feature Learning, was ran on Imagenet ILSVRC 2012 dataset to classify images. 

---

## Requirements
1. ILSVRC 2012 dataset should be downlaoded and  extracted under data/imagenet/
2. Python2
3. theano
4. pytorch
5. scipy
6. matplotlib
7. numpy
8. joblib
9. lazy_python
10. flask
11. itertools
12. OrderedDict
13. data
14. argparse
15. rescale
---

### To Run
  To train the model from strach, run train_imagenet.sh

  To evaluate the model, run eval_model.sh

