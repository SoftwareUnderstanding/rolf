
# 1. Introduction

Data scientists often choose the uniform distribution or the normal distribution as the latent variable distribution when they build representative models of datasets. For example, the studies of the GANs [1] and the VAEs[2] used the uniform random distribution and the normal one, respectively.

As the approximate function implemented by neural networks is usually continuous, the topological structure of the latent variable distribution is preserved after the transformation from the latent variables space to the observable variables space. Given that the observed variables are distributed on a torus and that networks, for example the GANs, are trained with the latent variables sampled from the normal distribution, the structure of the projected distribution by the trained networks does not meet with the torus, even if residual error is small enough. Imagine another example where the observable variables follow a mixture distribution, of which clusters separate each other, trained variational autoencoder can encode the feature on the latent variable space with high precision, however, the decoded distribution consists of connected clusters since the latent variable is topologically equal with the ball. This means that the topology of the given dataset is not represented by the projection of the trained networks.

In this short text, we study the consequence of autoencoders' training due to  the topological mismatch. We use the SAE[4] as autoencoders, which is enhanced based on the WAE[3] owing to the sinkhorn algorithm.

# 2. Specifications

## 2-1. Agent

Agents consist of encoder and decoder networks. The encoder networks transform observable variables into latent variables and the decoder networks reverse the latent variables into the represented observable variables.

In our case studies, we define these two networks as the multilayer perceptron with the following hyperparameters: number of units `nH`, number of layers `nLayer` and activation function `activation`, where the encoder and decoder networks share the same hyperparameters. The values of the hyperparameters are defined in each case study.

## 2-2. Environment

Environments generate datasets sampled from distributions of observable variables. The distributions are defined in each case study.

## 2-3. Trainer

In all the case studies, we adopt Sinkhorn AutoEncoder (SAE) objective[4] as training criteria. The regularization parameter `reg_param` is defined in each case study.


# 3. Case studies

# 3-1. Case study #1:

This case study builds representative models of a two-dimensional 1-torus by using the autoencoder with the latent variables sampled from the two dimensional uniform distribution. We show an example of the consequence caused by the topological mismatch between the observable and latent variables distribution.

Models are trained by using the hyperparameters shown in the table 3.1.1. The figure 3.1.1 (a) and (b) show the learning curves of the following training performances, respectively.
- Representative error, `mean((Y-Yhat)^2)`, where `Y and Yhat` are the original observed variables and the represented ones, respectively.
- Discrepancy between the referenced distribution of the latent variables and the projected ones by the trained encoder. Note that the discrepancy is measured by the absolute norm Wasserstein distance.

The learning curves tell us that the training has converged at the end of the training iterations.

The figure 3.1.2(a) (or the figure 3.1.2(b)) shows the images projected through the encoder (or decoder) of the trained model which has the average performance among the trained models with `nLayer=7`. The left one is the input image of the observed (or latent) variables approximated by an analytical function and the right one is obtained by projecting the input image via the trained encoder(or decoder). Here are our findings.

- The learning curves in the figure 3.1.1(a) and (b) tell us that the projected samples can match well with the original samples and the distribution of the latent variables looks like the uniform distribution.
- The figure 3.1.2(a) shows that the hole in the latent variables image is fairly small and that the referenced latent variables distribution is almost covered by the projected one.
- Seeing the figure 3.1.2(b), the decoder's projected image is topologically identified with the disk, even though the region around the hole is stretched.

The last two findings say that the encoder and decoder as maps between the observable variables and the latent ones cannot preserve the topological structure. This might cause practical problems. For example, if you optimize a function defined on a 1-torus and if you plan to parameterize the decision variables on the torus by using the latent variables defined by autoencoder, it might be possible that you find a solution at a point of the hole of the torus, which is of course infeasible, because it exists a certain area in the latent variable which can be mapped on the hole of the torus.


Table 3.1.1. Hyper parameters 

|name|description|value|
|-|-|-|
|nEpoch|the number of epochs | 512|
|nBatch|the sample size of a single batch|512|
|nH|the number of units of the encoder and decoder network|512|
|nLayer|the number of layers of the encoder and decoder network| 3, 5 and 7|
|reg_param|the regularization parameter | 10 |

<img src = "./casestudies/img/cs01b_representative_error.png" width = "50%"> 
Fig 3.1.1(a) Learning curve of the representative error grouped by the number of layers in the network

<img src = "./casestudies/img/cs01b_latent_distribution_discrepancy.png" width = "50%"> 
Fig 3.1.1(b) Learning curve of the discrepancy between the the referenced and the projected  latent variable distributions grouped by the number of layers in the network

<img src = "./casestudies/img/encoder_projection_cbarbaUWpfmwvNQS.png" width = "50%"> 
Fig 3.1.2(a) Input and output image of a trained encoder

<img src = "./casestudies/img/deccoder_projection_cbarbaUWpfmwvNQS.png" width = "50%"> 
Fig 3.1.2(b) Input and output image of a trained decoder

# 3-2. Case study #2:

We move on to the next case study to see another type of topological mismatch:
the one distributes on the twisted surface in the three dimensional space,
while the distribution of the other, not twisted.
It's impossible that the autoencoders consilliate this difference
since the twisted image (or not twisted) is mapped on to the twisted image (or not twisted).
We see the consequence of the autoencoders' training subject to this topological mismatch.

Here is the specifications of our experiment.
The environment generates the dataset sampled randomly from the mobius band.
More precisely say that the variables `x, y and z` in the three dimensional space randomly distribute on the surface defined in 
[site](https://en.wikipedia.org/wiki/M%C3%B6bius_strip#Geometry_and_topology)
.

On the other hand,
we define the agent that the distribution of the latent variables `u, v and w` follows
the uniform random distribution over a ring as follow:

<img src = "./casestudies/img/texclip20200826105519.png" width = "83%">

Note that the observable variables' distribution is twisted,
while the latent variables' one is not.

We train agents by using the hyperparameters in the table 3.2.1 and 
the figure 3.2.1 shows the learning curves of the pair of performances
mentioned already in the case study #1.
It tells us that the training has converged at the end of the final epoch.
We see below in detail an agent among trained agents around the average performance.

The figure 3.2.2(a) shows how the trained encoder maps the observable variable image
(the blue in the left) to the latent variable image (the blue one in the right).
The projected image on the latent variable space represents well the referenced image(the gray one).
Particularly, the part mapped from the twisted part of the input image 
is pushed and piled up on the surface of the referenced image 
due to the fact that the encoder cannot untangle the distortion of the mobius band.

The figure 3.2.2(b) shows the referenced latent variable image(the red one in the left) 
and its projected image on the observable variable space(the red one in the right)
by the trained decoder.
As mentioned in the case of the encoder, the projected image looks like the referenced observable image(the gray one in the right),
though, since the referenced latent image is not twisted, a part of projected image is stretched and flipped in to fit to the twisted part of the mobius band.
This happens because the decoder has to preserve the topological structure.

Thus, however well autoencoders regenerate datasets of distributions with complex structures
in the data-driven manner, they cannot represent topological structure.


Table 3.2.1. Hyper parameters 

|name|description|value|
|-|-|-|
|nEpoch|the number of epochs | 512|
|nBatch|the sample size of a single batch|512|
|nH|the number of units of the encoder and decoder network|128|
|nLayer|the number of layers of the encoder and decoder network|3|
|reg_param|the regularization parameter | 10 |

<img src = "./casestudies/img/cs02a_score.png" width = "50%"> 
Fig 3.2.1 Learning curve of the representative error and the discrepancy between the the referenced and the projected  latent variable distributions

<img src = "./casestudies/img/cs02a_encoder_projection_SQMQIE81fexBjY9p_azim=270.png" width = "75%"> 
Fig 3.2.2(a) Input and output image of a trained encoder

<img src = "./casestudies/img/cs02a_deccoder_projection_SQMQIE81fexBjY9p_azim=000.png" width = "75%"> 
Fig 3.2.2(b) Input and output image of a trained decoder


# 3-3. Case study #3:

In the third example, 
we take account of the difference of knots as an example of topological mismatch.
We try to represent a type of knot by transforming another type of knot
and we see what happens to the autoencoders' training due to this discrepancy.

Here is the configuration of agents and environments:
- The environment randomly samples values from [a trefoil knot](https://en.wikipedia.org/wiki/Trefoil_knot#Descriptions) in three-dimensional space.
- The latent variables of the agents are sampled from an unknot, namely a simple ring, in three-dimensional space.

The table 3.3.1 shows the hyperparameter set for the training.
Note that small batch size is required in this training, 
probably because a smaller size batch can break better the symmetry across the x-y plane which the observable variables distribution holds.
The training has already saturated at the end of epochs, which is confirmed in the learning curves shown in the figure 3.3.1.

We select one trained agent among the trained agents around average performances and analyze it.
- The figure 3.3.2(a) shows the transformation of the observable variables distribution
on the latent variables space by the trained encoder
Although the output image is close to the referenced latent variables distribution
, the output image seemingly preserves the knots of the original image.
- The figure 3.3.2(b) shows the referenced latent variables image, that is the simple circle,
and its projected image by the trained decoder on the observable variables space.
The output closed loop is approaching the original trefoil knot,
However, it's hard to fit it completely because the projected image cannot make new knots on their own.

In this way, even if the numerical evaluations of the error and the discrepancy of distributions are small,
the autoencoders are not capable of creating new knots.
The topological discrepancy cannot be resolved just by the autoencoders.

Table 3.3.1. Hyper parameters 

|name|description|value|
|-|-|-|
|nEpoch|the number of epochs | 512|
|nBatch|the sample size of a single batch|32|
|nH|the number of units of the encoder and decoder network|32|
|nLayer|the number of layers of the encoder and decoder network|3|
|reg_param|the regularization parameter | 0.1 |
|activation|activation function of agent|tanh|

<img src = "./casestudies/img/cs03c_score.png" width = "50%"> 
Fig 3.3.1 Learning curve of the representative error and the discrepancy between the the referenced and the projected  latent variable distributions

<img src = "./casestudies/img/cs03c_encoder_projection_A8h1YHFMvFCZkzb0_azim=000.png" width = "75%"> 
Fig 3.3.2(a) Input and output image of a trained encoder

<img src = "./casestudies/img/cs03c_decoder_projection_A8h1YHFMvFCZkzb0_azim=000.png" width = "75%"> 
Fig 3.3.2(b) Input and output image of a trained decoder


# 4. References

<ul>
<li>[1]:
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio: “Generative Adversarial Networks”, 2014; <a href='http://arxiv.org/abs/1406.2661'>arXiv:1406.2661</a>.
</li>
<li>[2]:
Diederik P Kingma, Max Welling: “Auto-Encoding Variational Bayes”, 2013; <a href='http://arxiv.org/abs/1312.6114'>arXiv:1312.6114</a>.
</li>
<li>[3]:
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, Bernhard Schoelkopf: “Wasserstein Auto-Encoders”, 2017; <a href='http://arxiv.org/abs/1711.01558'>arXiv:1711.01558</a>.
</li>
<li>[4]:
Giorgio Patrini, Rianne van den Berg, Patrick Forré, Marcello Carioni, Samarth Bhargav, Max Welling, Tim Genewein, Frank Nielsen: “Sinkhorn AutoEncoders”, 2018; <a href='http://arxiv.org/abs/1810.01118'>arXiv:1810.01118</a>.
</li>
</ul>
