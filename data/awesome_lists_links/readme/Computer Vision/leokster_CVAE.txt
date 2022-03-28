<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** leokster, CVAE, twitter_handle, email, Variational Autoencoder, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!--[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url1]
[![LinkedIn][linkedin-shield]][linkedin-url2]-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/leokster/CVAE">
    <img src="images/logo.png" alt="Logo" height="200">
  </a>

  <h3 align="center">Conditional Variational Autoencoder (CVAE)</h3>

  <p align="center">
    The conditional variational autoencoder is an extension of the classical autoencoder
    introduced by Kingma and Welling in 2014 [1]. The here proposed CVAE can be used in 
    any existing machine learning pipeline. One has to build the three models (see examples)
    Decoder, Encoder and Prior tailored to the data, the CVAE will then put them together
    into one single model. 
  
<br />
    <!--<a href="https://github.com/leokster/CVAE"><strong>Explore the docs »</strong></a>
    <br />-->
    <br />
    <a href="https://github.com/leokster/CVAE/tree/main/examples">View Demo</a>
    ·
    <a href="https://github.com/leokster/CVAE/issues">Report Bug</a>
    ·
    <a href="https://github.com/leokster/CVAE/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS 
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project"></a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

-->

<!-- ABOUT THE PROJECT -->
## Conditional Variational Autoencoder

If this is your first contact with such generative models one might note that the three individual networks
don't output esimations directly, all of them have three outputs. First the mean of a Gaussian, 
second the log variance of a Gaussian, and third a sample from the corresponding Gaussian. The
three Models are then chained as shown in the figure below. 

![conditional_variational_autoencoder][vae_architecture]

The lossfunction we use to train the model is the evidence lower bound (ELBO), which lower bounds the
posterior log-likelihood of the distribution ![posterior][posterior]. The loss function is then given
by ![total_loss_function] with ![dkl] and ![likl]

### P-Plot
It is difficult to evaluate models, which output whole distributions for every input.
Hence, one observes for every datapoint in the test set one distribuiton. We propose the
p-value evaluation method which is inspired from the Q-Q plots. For a list of empirical 
distributions (we call it H0) given by its samples and a list of one sample for each distribution, 
we compute the probability of obtain a sample at least as unlikely as our drawn sample is. If the
samples are drawn from H0 one would observe that these p-values are uniform (0,1)-distributed.

Compare the figure below, where the rows are the corresponding H0 distributions and the columns
determine from what distributions the samples are drawn. The blue lines in the (non histogram) 
figures shows the empirical cumulative distribution function of the observed p-values. The
gray area shows the deviance from the optimum (which is a straight line). 
![p_plot]

### Examples

#### MNIST
The VAE was trained to generate the MNIST dataset conditioned on the label (one-hot-encoded). 
In the figure below you can see some samples the VAE is producing.
![MNIST](images/mnist.png)


#### Power grid load data
This example shows how time series can be generated. As example we choose the public available
data from the Swiss transmission system operator (TSO) Swissgrid. The individual data samples
describe total electric load consumed in Switzerland in a resoultion of one hour. The figure below 
shows a true load profile (blue) and the outputs of the VAE (black dots). The VAE was fed with the 
last 24 hours of load (left part of the blue line). 
![MNIST](images/powergrid.png)


### Requirements

* Tensorflow
* Scikit-Learn 



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/leokster/CVAE.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```




<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/leokster/CVAE/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License.

<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/leokster/CVAE](https://github.com/leokster/CVAE)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Koen Van Walstijn
* Tim Rohner


## Literature
[1]   https://arxiv.org/abs/1312.6114



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/leokster/CVAE.svg?style=for-the-badge
[contributors-url]: https://github.com/leokster/CVAE/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/leokster/CVAE.svg?style=for-the-badge
[forks-url]: https://github.com/leokster/CVAE/network/members
[stars-shield]: https://img.shields.io/github/stars/leokster/CVAE.svg?style=for-the-badge
[stars-url]: https://github.com/leokster/CVAE/stargazers
[issues-shield]: https://img.shields.io/github/issues/leokster/CVAE.svg?style=for-the-badge
[issues-url]: https://github.com/leokster/CVAE/issues
[license-shield]: https://img.shields.io/github/license/leokster/CVAE.svg?style=for-the-badge
[license-url]: https://github.com/leokster/CVAE/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url1]: https://linkedin.com/in/tim-rohner
[linkedin-url2]: https://linkedin.com/in/koen-van-walstijn
[vae_architecture]: images/vae.png
[p_plot]: images/p_plt.png
[posterior]: https://chart.apis.google.com/chart?cht=tx&chl=\log%20p(y\mid%20x)
[total_loss_function]: https://chart.apis.google.com/chart?cht=tx&chl=%5Cmathcal%20L%20%3D0.5%5Cmathcal%20L_%7B%5Ctext%7Blikelihood%7D%7D-0.5%5Cmathcal%20L_%7BDKL%7D
[likl]: https://chart.apis.google.com/chart?cht=tx&chl=\mathcal%20L_{\text{likelihood}}=\frac{(y-\mu_d)^2}{\sigma_d^2}%2B\log\sigma_d^2
[dkl]: https://chart.apis.google.com/chart?cht=tx&chl=%5Cmathcal%20L_%7BDKL%7D%20%3D1%2B%5Clog%20%5Csigma%5E2%20-%20%5Clog%20%5Csigma_p%5E2%20-%20%5Cfrac%7B1%7D%7B%5Csigma_p%5E2%7D%5Cleft%5B%5Csigma%5E2%2B(%5Cmu-%5Cmu_p)%5E2]
