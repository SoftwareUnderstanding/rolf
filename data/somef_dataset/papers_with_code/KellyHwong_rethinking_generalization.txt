# rethinking_generalization

UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION

# install

1. Clone this repo
2. Follow install instructions from [tensorflow object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) and store in the resnet dirctory.
3. Clone fashion mnist [repo](https://github.com/zalandoresearch/fashion-mnist)

# visualizing results

```
tensorboard --logdir={path_to_this_directory}/summaries --port=5000
```

then go to 0.0.0.0:5000

tensorboard example:
![](./fig/image.png?raw=true)

# Conlusion

To summarize, using residual learning or rethinking generaliztion, is not aimed to train a network model for directly classification or some else learning and working purpose. It shows another approach to dig the learning ability of a parameterized network. This ability is reflected by forcing the training label totally shuffled from original classes. And by simple notification that some nearly identical training images are forcely tagged with different labels, or reversively speaking, that different labels are projected into very similar positions in pixel space, we can tell that such a residual network has such potention of residual learning ability.

# References

1. Fashion MNIST https://www.kaggle.com/zalando-research/fashionmnist
2. Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
3. Understanding deep learning requires rethinking generalization https://arxiv.org/abs/1611.03530
4. https://github.com/pluskid/fitting-random-labels
