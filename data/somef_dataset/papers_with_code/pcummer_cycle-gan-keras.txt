# cycle-gan-keras

A simple keras implementation of CycleGAN (https://arxiv.org/pdf/1703.10593.pdf) for unpaired image translation.

This implementation favors minor changes due to the generous application of skip connections throughout the generator network. The discriminator is also trained to distinguish real images of the two classes to avoid it learning to simply detect visual artifacts. 

Directions:
* Load images of one class into images_a and images of the second class into images_b
* Run the training routine to build the networks
* Run the testing routine to output translated versions of all images with discriminator confidences

A few tips:
* Narrower networks train more smoothly and are less prone to mode collapse; start at 8 nodes per layer
* Loss is not a great indicator of performance in adverserial setups; visualize the outputs periodically
* Keep the batch size for training very low or it will likely fail to learn

Examples:
Claude Monet <-> William Turner

We see that the model has learned to correctly add/remove texture, though it also learned a general pallete swap that is undesirable.

Turner (translated) <- Monet (original)

![Monet to Turner 1](output/Claude_Monet_24.jpg_translated_[[0.53027546]].png)
![Monet to Turner 2](output/Claude_Monet_71.jpg_translated_[[0.42348522]].png)

Monet (translated) <- Turner (original)

![Turner to Monet 1](output/William_Turner_53.jpg_translated_[[1.]].png)
![Turner to Monet 2](output/William_Turner_6.jpg_translated_[[0.6505608]].png)
