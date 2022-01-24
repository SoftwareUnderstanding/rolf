# Recurrence and Attention in Latent Variable Models
In recent years deep latent variable models have been widely used for image generation and representation learning. Standard approaches employ shallow inference models with restrictive mean-field assumptions.  A way to increase inference expressivity is to define a hierarchy of latent variables in space and build structured approximations. Using this approach the size of the model grows linearly in the number of layers.

An orthogonal approach is to define hierarchies in time using a recurrent model. This approach exploits parameter sharing and gives us the possibility to define models with infinite depth (assuming a memory-efficient learning algorithm).

In this project, we study recurrent latent variable models for image generation. We focus on attentive models, i.e. models that use attention to decide where to focus on and what to update, refining their output with a sequential update. This is done by implementing the DRAW model, which is described in [the DRAW paper](https://arxiv.org/abs/1502.04623), both with basic and filterbank attention. The performance of the implemented DRAW model is then compared to both a standard VAE and a LadderVAE implementation.

The project is carried out by [Simon Amtoft Pedersen](https://github.com/simonamtoft), and supervised by [Giorgio Giannone](https://georgosgeorgos.github.io/).

## Variational Autoencoder
Variational Autoencoders (VAEs) are a type of latent variable model that can be used for generative modelling. The VAEs consists of a decoder part and an encoder part, that is trained by optimizing the Evidence Lower Bound (ELBO). The generative model is given by <img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z)&space;=&space;p_\theta(x|z)&space;p_\theta(z)" title="\inline p_\theta(z) = p_\theta(x|z) p_\theta(z)" /> and the samples are then drawn from the distribution by <img src="https://latex.codecogs.com/svg.image?z\sim&space;p_\theta(z)" title="z\sim p_\theta(z)" /> and <img src="https://latex.codecogs.com/svg.image?x\sim&space;p_\theta(x|z)" title="x\sim p_\theta(x|z)" />, and reconstruction is drawn from <img src="https://latex.codecogs.com/svg.image?q_\phi(z|x)" title="q_\phi(z|x)" />. The objective is then to optimize <img src="https://latex.codecogs.com/svg.image?\inline&space;\sum_i&space;\mathcal{L_{\theta,\phi}}(x_i)" title="\inline \sum_i \mathcal{L_{\theta,\phi}}(x_i)" /> where ELBO is given as <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathcal{L_{\theta,\phi}}(x)&space;=&space;\mathbb{E}_{q_\phi(z|x)}[\log&space;p_\theta&space;(x|z)]&space;&plus;&space;\mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(z)}{q_\phi(z|x)}\right]" title="\inline \mathcal{L_{\theta,\phi}}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta (x|z)] + \mathbb{E}_{q_\phi(z|x)}\left[\log\frac{p_\theta(z)}{q_\phi(z|x)}\right]" />.

Once the model is trained, it can generate new examples by sampling <img src="https://latex.codecogs.com/svg.image?\inline&space;z&space;\sim&space;N(z|0,1)" title="\inline z \sim N(z|0,1)" /> and then passing this sample through the decoder to generate a new example <img src="https://latex.codecogs.com/svg.image?\inline&space;x&space;\sim&space;N(x|\mu(z),&space;diag(\sigma^2(z)))" title="\inline x \sim N(x|\mu(z), diag(\sigma^2(z)))" />.


### Ladder VAE
An extension of the standard VAE is the [Ladder VAE](https://arxiv.org/pdf/1602.02282.pdf), which adds sharing of information and parameters between the encoder and decoder by splitting the latent variables into L layers, such that the model can be described by:

<img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z)&space;=&space;p_\theta(z_L)\prod_{i=1}^{L-1}&space;p_\theta(z_i&space;|z_{i&plus;1})&space;" title="\inline p_\theta(z) = p_\theta(z_L)\prod_{i=1}^{L-1} p_\theta(z_i |z_{i+1}) " />

<img src="https://latex.codecogs.com/svg.image?\inline&space;p_\theta(z_i&space;|&space;z_{i&plus;1})&space;=&space;N(z_i|&space;\mu_{p,i},&space;\sigma^2_{p,i}),&space;\;\;\;\;&space;p_\theta(z_L)&space;=&space;N(z_L|0,I)" title="\inline p_\theta(z_i | z_{i+1}) = N(z_i| \mu_{p,i}, \sigma^2_{p,i}), \;\;\;\; p_\theta(Z_L) = N(z_L|0,I)" />

<img src="https://latex.codecogs.com/svg.image?p_\theta(x|z_1)&space;=&space;N(x|\mu_{p,0},\sigma^2_{p,0})" title="p_\theta(x|z_1) = N(x|\mu_{p,0},\sigma^2_{p,0})" />

A lot of the code for the Ladder VAE is taken from [Wohlert semi-supervised pytorch project](https://github.com/wohlert/semi-supervised-pytorch).

## Deep Recurrent Attentive Writer
The Deep Recurrent Attentive Writer (DRAW) model is a VAE like model, trained with stochastic gradient descent, proposed in the [original DRAW paper](https://arxiv.org/pdf/1502.04623.pdf). The main difference is, that the DRAW model iteratively generates the final output instead of doing it in a single shot like a standard VAE. Additionally, the encoder and decoder uses recurrent networks instead of standard linear networks.

### The Network
The model goes through T iterations, where we denote each time-step iteration by t. When using a diagonal Gaussian for the latent distribution, we have:

<img src="https://latex.codecogs.com/svg.image?\mu_t&space;=&space;W(h_t^{enc}),&space;\;\;\;&space;\;\;&space;\sigma_t&space;=&space;\exp(W(h_t^{enc}))" title="\mu_t = W(h_t^{enc}), \;\;\; \;\; \sigma_t = \exp(W(h_t^{enc}))" />

Samples are then drawn from the latent distribution <img src="https://latex.codecogs.com/svg.image?z_t&space;\sim&space;Q(z_t|h_t^{enc})" title="z_t \sim Q(z_t|h_t^{enc})" />, which we pass to the decoder, which outputs <img src="https://latex.codecogs.com/svg.image?h_t^{dec}" title="h_t^{dec}" /> that is added to the canvas, <img src="https://latex.codecogs.com/svg.image?c_t" title="c_t" />, using the write operation. At each time-step, <img src="https://latex.codecogs.com/svg.image?t&space;=&space;1,...,T" title="t = 1,...,T" />, we compute:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!&space;\hat{x}_t&space;=&space;x&space;-&space;\sigma(c_{t-1})\\r_t&space;=&space;read(x_t,\hat{x}_t,h_{t-1}^{dec})\\h_t^{enc}&space;=&space;RNN^{enc}(h_{t-1}^{enc},&space;[r_t,&space;h_{t-1}^{dec}]])\\z_t&space;\sim&space;Q(z_t|h_t^{enc})\\h_t^{dec}&space;=&space;RNN^{dec}(h_{t-1}^{dec},&space;z_t)\\c_t&space;=&space;c_{t-1}&space;&plus;&space;write(h_t^{dec})&space;" title="\!\!\!\!\!\!\!\!\! \hat{x}_t = x - \sigma(c_{t-1})\\r_t = read(x_t,\hat{x}_t,h_{t-1}^{dec})\\h_t^{enc} = RNN^{enc}(h_{t-1}^{enc}, [r_t, h_{t-1}^{dec}]])\\z_t \sim Q(z_t|h_t^{enc})\\h_t^{dec} = RNN^{dec}(h_{t-1}^{dec}, z_t)\\c_t = c_{t-1} + write(h_t^{dec}) " />

### Data Generation
Generating images from the model is then done by iteratively picking latent samples from the prior distribution, and updating the canvas with the decoder:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!\tilde{z}_t&space;\sim&space;p(z_t)\\\tilde{h}_t^{dec}&space;=&space;RNN^{dec}(\tilde{h}_{t-1}^{dec},\tilde{z})\\\tilde{c}_t&space;=&space;\tilde{c}_{t-1}&space;&plus;&space;write(\tilde{h}_t^{dec})\\\tilde{x}&space;\sim&space;D(X|\tilde{c}_T)&space;" title="\!\!\!\!\!\!\!\!\!\tilde{z}_t \sim p(z_t)\\\tilde{h}_t^{dec} = RNN^{dec}(\tilde{h}_{t-1}^{dec},\tilde{z})\\\tilde{c}_t = \tilde{c}_{t-1} + write(\tilde{h}_t^{dec})\\\tilde{x} \sim D(X|\tilde{c}_T) " />

### Read and Write operations
Finally we have the read and write operations. These can be used both with and without attention.

In the version without attention, the entire input image is passed to the encoder for every time-step, and the decoder modifies the entire canvas at every step. The two operations are then given by

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!read(x,&space;\hat{x}_t,&space;h_{t-1}^{dec})&space;=&space;[x,&space;\hat{x}_t]\\write(h_t^{dec})&space;=&space;W(h_t^{dec})&space;" title="\!\!\!\!\!\!\!\!read(x, \hat{x}_t, h_{t-1}^{dec}) = [x, \hat{x}_t]\\write(h_t^{dec}) = W(h_t^{dec}) " />

In oder to use attention when reading and writing, a two-dimensional attention form is used with an array of two-dimensional Gaussian filters. For an input of size A x B, the model generates five parameters from the output of the decoder, which is used to compute the grid center, stride and mean location of the filters:

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!(\tilde{g}_X,&space;\tilde{g}_Y,&space;\log&space;\sigma^2,&space;\log&space;\tilde{\delta},&space;\log&space;\gamma)&space;=&space;W(h^{dec}_t)\\g_X&space;=&space;\frac{A&plus;1}{2}(\tilde{g}_X&space;&plus;&space;1)\\g_X&space;=&space;\frac{A&plus;1}{2}(\tilde{g}_X&space;&plus;&space;1)\\&space;\delta&space;=&space;\frac{\max(A,B)&space;-&space;1}{N&space;-&space;1}&space;\tilde{\delta}\\\mu_X^i&space;=&space;g_X&space;&plus;&space;(i&space;-&space;N/2&space;-&space;0.5)&space;\delta\\\mu_Y^j&space;=&space;g_Y&space;&plus;&space;(j&space;-&space;N/2&space;-&space;0.5)&space;\delta&space;" title="\!\!\!\!\!\!\!\!\!(\tilde{g}_X, \tilde{g}_Y, \log \sigma^2, \log \tilde{\delta}, \log \gamma) = W(h^{dec}_t)\\g_X = \frac{A+1}{2}(\tilde{g}_X + 1)\\g_X = \frac{A+1}{2}(\tilde{g}_X + 1)\\ \delta = \frac{\max(A,B) - 1}{N - 1} \tilde{\delta}\\\mu_X^i = g_X + (i - N/2 - 0.5) \delta\\\mu_Y^j = g_Y + (j - N/2 - 0.5) \delta " />

From this, the horizontal and veritcal filterbank matrices is defined

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!F_X[i,a]&space;=&space;\frac{1}{Z_X}&space;\exp\left(-\frac{(a-\mu_X^i)^2}{2\sigma^2}\right)\\F_Y[j,b]&space;=&space;\frac{1}{Z_Y}&space;\exp\left(-\frac{(b-\mu_Y^i)^2}{2\sigma^2}\right)&space;" title="\!\!\!\!\!\!\!\!\!F_X[i,a] = \frac{1}{Z_X} \exp\left(-\frac{(a-\mu_X^i)^2}{2\sigma^2}\right)\\F_Y[j,b] = \frac{1}{Z_Y} \exp\left(-\frac{(b-\mu_Y^i)^2}{2\sigma^2}\right) " />

Where <img src="https://latex.codecogs.com/svg.image?Z_X" title="Z_X" /> and <img src="https://latex.codecogs.com/svg.image?Z_Y" title="Z_Y" /> are normalisation constraints, such that 

<img src="https://latex.codecogs.com/svg.image?\sum_a&space;F_X[i,a]&space;=&space;1" title="\sum_a F_X[i,a] = 1" /> 

<img src="https://latex.codecogs.com/svg.image?\sum_b&space;F_Y[j,b]&space;=&space;1" title="\sum_b F_Y[j,b] = 1" />

Finally, we define the read and write operations with the attention mechanism

<img src="https://latex.codecogs.com/svg.image?\!\!\!\!\!\!\!\!\!read(x,&space;\hat{x}_t,&space;h_{t-1}^{dec})&space;=&space;\gamma[F_Y&space;x&space;F_X^T,&space;F_Y&space;\hat{x}&space;F_X^T]\\w_t&space;=&space;W(h_t^{dec})\\write(h_t^{dec})&space;=&space;\frac{1}{\hat{\gamma}}&space;\hat{F}_Y^T&space;w_t&space;\hat{F}_X&space;" title="\!\!\!\!\!\!\!\!\!read(x, \hat{x}_t, h_{t-1}^{dec}) = \gamma[F_Y x F_X^T, F_Y \hat{x} F_X^T]\\w_t = W(h_t^{dec})\\write(h_t^{dec}) = \frac{1}{\hat{\gamma}} \hat{F}_Y^T w_t \hat{F}_X " />

Where <img src="https://latex.codecogs.com/svg.image?w_t" title="w_t" /> is the N x N writing patch emitted by the decoder.

## Results
The standard VAE, Ladder VAE and DRAW model with base attention have been trained on the standard [torchvision MNIST dataset](https://pytorch.org/vision/stable/datasets.html), which is transformed to be in binarized form. Below the final value of the ELBO, KL and Reconstruction metrics are reported for both the train, validation and test set. Additionally the loss plots for training and validation is shown, and finally some reconstruction and samples from the three different models are shown.

The three models are trained in the exact same manner, without using a lot of tricks to improve upon their results. For all models KL-annealing is used over the first 50 epochs. Additionally, every model uses learning rate decay that starts after 200 epochs and is around halved at the end of training. To check the model parameters used, inspect the config dict in the three different training files.

### Loss Plots
![alt text](https://github.com/simonamtoft/ml-library/blob/main/results/loss%20plots.png?raw=true)

### Table Evaluations
|Train              | ELBO    | KL    | Reconstruction |
| --- | --- | --- | --- |
|Standard VAE       | -121.88 | 25.96 | 95.92|
|Ladder VAE         | -123.75 | 25.18 | 98.57|
|DRAW Base Attention| -84.04  | 25.81 | 58.22|

|Validation         | ELBO      | KL    | Reconstruction |
| --- | --- | --- | --- |
|Standard VAE       | -124.38   | 25.93 | 98.45 |
|Ladder VAE         | -124.12   | 24.98 | 99.14 |
|DRAW Base Attention| -85.43    | 25.88 | 59.55 |

|Test               | ELBO    | KL    | Reconstruction |
| --- | --- | --- | --- |
|Standard VAE       | -123.81 | 25.13 | 98.68|
|Ladder VAE         | -123.43 | 25.06 | 98.37|
|DRAW Base Attention| -85.3   | 26.01 | 59.28|

### Samples and Reconstructions
![alt text](https://github.com/simonamtoft/ml-library/blob/main/results/images.png?raw=true)

## Discussion
From the results it is clear that the implemented DRAW model performs better than the standard VAE. However, it is also seen that the Ladder VAE has kind of collapsed into the standard VAE, providing far worse results than in the original paper. This can be due to multiple things. First of all the model might not be exactly identical to the proposed model regarding the implementation itself and number and size of layers. Secondly, all the three models are trained in exactly the same manner, without using a lot of tricks to improve the training of the Ladder VAE, which was done in the paper.

### Comment on DRAW with Filterbank Attention
The filterbank attention version of the DRAW model is somewhat of a work-in-progress. It [seems to be implemented correctly](https://github.com/simonamtoft/ml-library/blob/main/notebooks/A%20Look%20at%20Attention.ipynb) using a batch size of one, but very slow computationally. Additionally when running only with a batch size of one, each epoch takes too long to make it feasible. In order to make this model able to work in practice one would have to optimize it for batch sizes larger than one and improve the computational speed.

## Repo Structure
In this repo you will find the three different model classes in the models directory, and the necessary training loops for each model is found in the training directory.
Additionally the attention, encoder and decoder, and other modules used in these models can be found in the layers directory.

In order to reproduce the results, first make sure you have torch installed and all the required packages specified in `requirements.txt`, it should then be fairly simple to run the training of each of the models by using the appropriate script: `python run_vae.py` or `python run_lvae.py` or `python run_draw.py`. In order to change any model or training parameters, simply change the config dict inside these scripts.

## References
Diederik P. Kingma & Max Welling: An Introduction to Variational Autoencoders, [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)

Carl Doersch: Tutorial on Variational Autoencoders, [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)

Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby & Ole Winther, Ladder Variational Autoencoders, [arXiv:1602.02282](https://arxiv.org/abs/1602.02282)

Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende & Daan Wierstra: DRAW A Recurrent Neural Network For Image Generation, [arXiv:1502.04623](https://arxiv.org/abs/1502.04623)



