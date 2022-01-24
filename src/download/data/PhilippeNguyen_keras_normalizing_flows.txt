# keras_normalizing_flows
Keras implementation of the two normalizing flows introduced by Rezende/Mohamed 2016 (https://arxiv.org/abs/1505.05770).

The flows are implemented as keras layers.

This was mostly instructional for me, I quickly found tensorflow_probability is much better suited for this kind of thing.

Please send me any questions/corrections.
#### Requirements
Only keras is really needed to use the layers.

However, tensorflow (tensorflow_probability), matplotlib and numpy as required for the scripts

#### Scripts
Two of the scripts (planar_logprob.py and radial_logprob.py) are showing the transformation using 1 flow. Of course, these are cooked simple examples, to get more complex distributions, you'll need flow stacks and more complex transformations.

Planar Flow (unimodal gaussian to multimodal gaussian)

|  |  |  |
| --- | --- | --- |
| ![planar_logprob_target](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/planar_logprob_target.png "planar_logprob_target") | ![planar_logprob_base](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/planar_logprob_base.png "planar_logprob_base") | ![planar_logprob_transformed](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/planar_logprob_transformed.png "planar_logprob_transformed") |

Radial Flow (uniform to gaussian-ish)

|  |  |  |
| --- | --- | --- |
| ![radial_logprob_target](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/radial_logprob_target.png "radial_logprob_target") | ![radial_logprob_base](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/radial_logprob_base.png "radial_logprob_base") | ![radial_logprob_transformed](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/radial_logprob_transformed.png "radial_logprob_transformed") |

Last script shows a stack of flows, transforming the usual isotropic gaussian latent into something more complex before sending it along to a non-flow NN decoder. Generally, in more sophisticated applications, the decoder is represented as a flow too.

|  |  |
| --- | --- |
| ![mnist_base](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/mnist_base.png "mnist_base") | ![mnist_transformed](https://github.com/PhilippeNguyen/keras_normalizing_flows/blob/master/assets/mnist_transformed.png "mnist_transformed") |
