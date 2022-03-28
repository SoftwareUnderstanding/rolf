Conditional Generative Adversarial Networks
===========================================

![Demonstration of deterministic control of image samples. We tweak conditional information to first make the sampled faces age, then again to make them smile.](https://hans.github.io/uploads/2015/conditional-gans-face-generation/axis_incremental.png)

The code in this repository implements the conditional generative
adversarial network (cGAN), described in my paper from late 2015:

[*Conditional generative adversarial networks for convolutional face
generation.*][1] Jon Gauthier. March 2015.

This code is a fork of the [original GAN repository][2]. The original
GAN model is described in the paper:

[*Generative Adversarial Networks.*][3] Ian J. Goodfellow, Jean Pouget-Abadie,
Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
Yoshua Bengio. ArXiv 2014.

## Guide to the code / usage instructions

This code is built on the [Pylearn2][4] framework for machine learning.
The abstract model structures are implemented as Python classes (see e.g.
the [`ConditionalAdversaryPair` class][5], and concrete model
instantiations / training configurations are described in YAML files
(see e.g. [a file for training with LFW data][6]).

You can perform your own training runs using these YAML files. The paths in
the YAML files reference my own local data; you'll need to download the LFW
dataset and change these paths yourself. The "file-list" and embedding files
referenced in the YAML files are available for LFW
[in the `data/lfwcrop_color` folder][7]. Once you have the paths in the YAML
file, you can start training a model with the simple invocation of Pylearn2's
`train.py` binary, e.g.

    train.py models/lfwcrop_convolutional_conditional.yaml
    
### Visualizations

The `sampler` folder contains various GAN sampling scripts that helps visualize
trained models. Some highlights are listed below (see the head of the linked
source files for descriptions).

- [`data_browser`][9]
- [`noise_browser`][8]
- [`show_samples_lfw_conditional`][10]
    
## Requirements

- Numpy
- Theano
- Pylearn2

[1]: https://github.com/hans/adversarial/blob/master/paper.pdf
[2]: https://github.com/goodfeli/adversarial
[3]: http://arxiv.org/abs/1406.2661
[4]: http://deeplearning.net/software/pylearn2/
[5]: https://github.com/hans/adversarial/blob/master/conditional/__init__.py#L13
[6]: https://github.com/hans/adversarial/blob/master/models/lfwcrop_convolutional_conditional_retrain.yaml
[7]: https://github.com/hans/adversarial/tree/master/data/lfwcrop_color
[8]: https://github.com/hans/adversarial/blob/master/sampler/noise_browser.py
[9]: https://github.com/hans/adversarial/blob/master/sampler/data_browser.py
[10]: https://github.com/hans/adversarial/blob/master/sampler/show_samples_lfw_conditional.py
