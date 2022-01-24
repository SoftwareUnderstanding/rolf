# h-vae
Project based on [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328).

## Used papers
Importance Weighted Autoencoder - [https://arxiv.org/abs/1509.00519](https://arxiv.org/abs/1509.00519);
Planar Normalizing Flows - [https://arxiv.org/abs/1505.05770](https://arxiv.org/abs/1505.05770);
Inverse Autoregressive Flows - [https://arxiv.org/abs/1606.04934](https://arxiv.org/abs/1606.04934).

## Project proposal
Link to the project proposal - [https://drive.google.com/file/d/1-q50kvccrze68GvE1DEoaRx2Kq_54LeG/view?usp=sharing](https://drive.google.com/file/d/1-q50kvccrze68GvE1DEoaRx2Kq_54LeG/view?usp=sharing).

## Requirements

We have run the experiments on Linux. The versions are given in brackets. The following packages are used in the implementation:
* [PyTorch (1.4.0)](https://pytorch.org/get-started/locally/)
* [NumPy (1.17.3)](https://numpy.org/)
* [SciPy (1.5.3)](https://docs.scipy.org/doc/)
* [scikit-learn (0.22.1)](https://scikit-learn.org/stable/)
* [matplotlib (3.1.2)](https://matplotlib.org/)
* [tqdm (4.39.0)](https://github.com/tqdm/tqdm)
* [Pyro (1.3.1)](https://pyro.ai/)


You can use [`pip`](https://pip.pypa.io/en/stable/) or [`conda`](https://docs.conda.io/en/latest/) to install them. 

## Contents

All experiments can be found in the underlying notebooks:

| Notebook      | Description |
|-----------|------------|
|[demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb) | **HVAE vs IWAE on MNIST:** experiments with HMC, training of the Hamiltonian Variational Auto-Encoder and Importance Weighted Autoencoder, reconstruction of encoded images, comparison of HMC trajectories.|
|[demos/hvae_gaussian_dim_(25/50/100/200/300/400).ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/hvae_gaussian_dim_300.ipynb) | **HVAE vs PNF/IAF/VB on Gaussian Model:** experiments with learning Hamiltonian Variational Auto-Encoder, Planar Normalizing Flows, Inverse Autoregressive Normalizing Flows, Variational Bayes for Gaussian Model in [https://arxiv.org/abs/1805.11328](https://arxiv.org/abs/1805.11328), comparison of learned <img src="svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/> and <img src="svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.87217899999999pt height=22.465723500000017pt/> parameters for all methods, comparison of learning processes. Number in the name of notebooks denotes the dimensionality <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/> for the problem.|

For convenience, we have also implemented a framework and located it correspondingly in [gaussians/api](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/gaussians/api).

## Results

### Gaussian Model
We compared all discussed methods for dimensions <img src="svgs/f67096da04471c9f50e31b00f7f50c14.svg?invert_in_darkmode" align=middle width=198.5103615pt height=22.831056599999986pt/>. Authors trained their models using optimization process for the whole dataset, but we found that HVAE results are better and training process is faster when the dataset is divided on batches. HVAE and normalizing flows were trained for <img src="svgs/946a7aaf620371ac3590184a18ac92c1.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> iterations across dataset divided on batches with <img src="svgs/9684129ebb778f48019391de80875252.svg?invert_in_darkmode" align=middle width=24.657628049999992pt height=21.18721440000001pt/> samples. For all experiments the dataset has <img src="svgs/28326d3ee086205259a55f1263e21783.svg?invert_in_darkmode" align=middle width=85.31952989999999pt height=22.465723500000017pt/> points and training was done using RMSProp with a learning rate of <img src="svgs/7478f3ddcc5c4a0d602772a3057efe42.svg?invert_in_darkmode" align=middle width=33.26498669999999pt height=26.76175259999998pt/> and were conducted with fix random seed = <img src="svgs/66598bc181ac25cca9c745e3ed395aec.svg?invert_in_darkmode" align=middle width=41.09604674999999pt height=21.18721440000001pt/>. We average the results for predicted <img src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> for <img src="svgs/5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> different generated datasets according to Gaussian model and present the mean results in the following figures: 
<p align="center">
  <img width="500" alt="Comparison of learned average theta" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_theta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learned average delta" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_delta_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learned average sigma" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/all_sigma_comparison.jpg?raw=true">
</p>

<p align="center">
  <img width="500" alt="Comparison of learning processes for different losses" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/gaussian_learning_processes.jpg?raw=true">
</p>

We trained Variational Bayes for big dimensions <img src="svgs/c822ed4d73bf20944683878604a747dd.svg?invert_in_darkmode" align=middle width=55.13122229999999pt height=22.831056599999986pt/> more iterations (<img src="svgs/124a322b51f8b8228ecd83a0a70c995d.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> or <img src="svgs/fcaa39039914d95fd04b3a004da93bf2.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/>) due to the fact that <img src="svgs/946a7aaf620371ac3590184a18ac92c1.svg?invert_in_darkmode" align=middle width=32.876837399999985pt height=21.18721440000001pt/> iterations were not enough for the convergence of ELBO. HVAE with tempering and IAF have the best learned <img src="svgs/494429435f09e010cd0f9ba62ffc7c59.svg?invert_in_darkmode" align=middle width=81.18702734999998pt height=31.50689519999998pt/> for the big dimensionality <img src="svgs/c822ed4d73bf20944683878604a747dd.svg?invert_in_darkmode" align=middle width=55.13122229999999pt height=22.831056599999986pt/>. Moreover, HVAE is good for <img src="svgs/c5f57ca07cdaed981ff201596a66a1b3.svg?invert_in_darkmode" align=middle width=13.652895299999988pt height=22.55708729999998pt/> prediction for all dimensions as well Variational Bayes scheme. However, Variationl Bayes suffers most on prediction <img src="svgs/7e9fe18dc67705c858c077c5ee292ab4.svg?invert_in_darkmode" align=middle width=13.69867124999999pt height=22.465723500000017pt/> as the dimension increases. Planar normalizing flows suffer on prediction <img src="svgs/c5f57ca07cdaed981ff201596a66a1b3.svg?invert_in_darkmode" align=middle width=13.652895299999988pt height=22.55708729999998pt/> compared to IAF. 

Also we compare HVAE with tempering and without tempering, see figure:
<p align="center">
  <img width="500" alt="Comparison across HVAE with and without tempering of learning theta, log-scale" src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_theta_comparison.jpg?raw=true">
</p>
We can see that the tempered methods perform better than their non-tempered counterparts; this shows that time-inhomogeneous dynamics are a key ingredient in the effectiveness of the method. 

### MNIST
We appeal to the binarized MNIST handwritten digit dataset as an example of image generative task. The training data has the following form: <img src="svgs/20be1f89af8faf288134c641cd6457f0.svg?invert_in_darkmode" align=middle width=126.65501144999998pt height=24.65753399999998pt/>, where <img src="svgs/e4e7063d5b30290b842d129a00adb48a.svg?invert_in_darkmode" align=middle width=118.03481414999999pt height=27.91243950000002pt/> for <img src="svgs/3f7803be2c3909174bb020625fa49416.svg?invert_in_darkmode" align=middle width=130.01688149999998pt height=22.831056599999986pt/>. We then formalize the generative model:
<p align="center"><img src="svgs/677679c03782e64805b57fe523db672d.svg?invert_in_darkmode" align=middle width=284.99496465pt height=76.6371606pt/></p>

for <img src="svgs/033a8bb0a13924a7a598a6c31f212805.svg?invert_in_darkmode" align=middle width=54.45300464999998pt height=24.65753399999998pt/> where <img src="svgs/f01f384acd603a2d28c8932c72dcaa5e.svg?invert_in_darkmode" align=middle width=33.75769979999999pt height=24.65753399999998pt/> is the <img src="svgs/8c22d0554871826c8a55e942afc8de77.svg?invert_in_darkmode" align=middle width=20.37223649999999pt height=27.91243950000002pt/> component of <img src="svgs/9afd941e1aa92a80c5e1020b83603408.svg?invert_in_darkmode" align=middle width=107.8936023pt height=27.91243950000002pt/>  is the latent variable associated with <img src="svgs/2d7ce8530d5d49ab97401227716d06a2.svg?invert_in_darkmode" align=middle width=19.43400854999999pt height=14.15524440000002pt/> and <img src="svgs/ef35c573a26a51f7572409522ddca01d.svg?invert_in_darkmode" align=middle width=83.42815964999998pt height=22.465723500000017pt/> is an encoder (convolutional neural network). The VAE approximate posterior is given by <img src="svgs/224d589b43fb1b6a10186a90dff24410.svg?invert_in_darkmode" align=middle width=283.10784315pt height=24.65753399999998pt/> where <img src="svgs/f699e14d6ebfdd5579d6918ccdad8735.svg?invert_in_darkmode" align=middle width=17.809057199999987pt height=14.15524440000002pt/> and <img src="svgs/ea53c3dff417c6a0be50f4943cf139ff.svg?invert_in_darkmode" align=middle width=19.77631259999999pt height=22.465723500000017pt/> are separate outputs of the encoder parametrized by <img src="svgs/68045bb2eaed78dc337918c3764f0891.svg?invert_in_darkmode" align=middle width=14.360768399999989pt height=22.831056599999986pt/> and <img src="svgs/ea53c3dff417c6a0be50f4943cf139ff.svg?invert_in_darkmode" align=middle width=19.77631259999999pt height=22.465723500000017pt/> is constrained to be diagonal. 

We set the dimensionalty of the latent space to <img src="svgs/2770c8d006c6b70701d31bd8c3cd78d6.svg?invert_in_darkmode" align=middle width=43.584393599999984pt height=22.831056599999986pt/>. As we need both means and variances to parametrize the VAE posterior distribution, the output dimension of the linear layer is set to <img src="svgs/9ed7c0348942d04eebcdb96fe0e91e53.svg?invert_in_darkmode" align=middle width=72.22931924999999pt height=22.831056599999986pt/>. We use Adam optimizer with standard parameters and learning rate set to <img src="svgs/7478f3ddcc5c4a0d602772a3057efe42.svg?invert_in_darkmode" align=middle width=33.26498669999999pt height=26.76175259999998pt/>. 

We compare the performance of HVAE with the performance of IWAE [https://arxiv.org/abs/1509.00519](https://arxiv.org/abs/1509.00519). We set the number of Monte-Carlo steps in HVAE and the number of importance samples in IWAE so that <img src="svgs/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> training epoch requiers equal time to finish. In this setting we can compare them more fairly. We fix the number Monte-Carlo steps <img src="svgs/12b0b94127c5535fadb5b3819e80b5a7.svg?invert_in_darkmode" align=middle width=74.34933989999999pt height=22.465723500000017pt/> and the number of importance samples <img src="svgs/c1d2fd0730cd454946fac68d9b9661e9.svg?invert_in_darkmode" align=middle width=71.13017009999999pt height=22.465723500000017pt/>. Both models are then optimized for <img src="svgs/2b797fbe6e7fa9936c37484c304423cc.svg?invert_in_darkmode" align=middle width=16.438418699999993pt height=21.18721440000001pt/> epochs. Corresponding plots are the following: 
<p align="center">
  <img width="500" alt="The training of HVAE and IWAE models." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_iwae_elbo.jpg?raw=true">
</p>

It can be clearly seen that the training loss values are similar for both models, while the validation loss of HVAE is higher due to overfitting. 

It is also important to compare the models outputs in terms of quality. The generated images are shown in the following figures (top - HVAE, bottom - IWAE): 
<p align="center">
  <img width="500" alt="Images generated by both models trained on the binarized MNIST dataset. HVAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/hvae_mnist.png?raw=true">
</p>
<p align="center">
  <img width="500" alt="Images generated by both models trained on the binarized MNIST dataset. IWAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/iwae_mnist.png?raw=true">
</p>
Images generated via IWAE appear to be more blured. However, HVAE tends to generate sharper images while some of them can not be recognized as digits. 

To better understand the behavior of both models, we study the decoded latent vectors of HMC chains (top - HVAE, bottom - IWAE):
<p align="center">
  <img width="500" alt="Trajectories of HMC given a latent vector corresponding to the source image. HVAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/steps_hvae_cropped.png?raw=true">
</p>
<p align="center">
  <img width="500" alt="Trajectories of HMC given a latent vector corresponding to the source image. IWAE." src="https://github.com/Daniil-Selikhanovych/h-vae/blob/master/images/steps_iwae_cropped.png?raw=true">
</p>

In these figures one can clearly see that HVAE encoded vectors often correspond to the class that is different from the ground-truth, even though they are sharper. At the same time, IWAE produces reconstructions that are close to the true images. 

### Details
More details about experiments and settings can be found in the report [https://github.com/Daniil-Selikhanovych/h-vae/blob/master/report/hvae_report.pdf](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/report/hvae_report.pdf).

## Pretrained Models

| Models      | Description |
|-----------|------------|
|[HVAE & IWAE on MNIST](https://drive.google.com/drive/folders/18KuruFMjmGfgyt_km747P4QuDYtVbJec?usp=sharing) |models from experiments in [demos/mnist.ipynb](https://github.com/Daniil-Selikhanovych/h-vae/blob/master/demos/mnist.ipynb)|

## LaTex in Readme

We have used [`readme2tex`](https://github.com/leegao/readme2tex) to render LaTex code in this Readme. Install the corresponding hook and change the command to fix the issue with broken paths:
```bash
python -m readme2tex --output README.md README.tex.md  --branch master --svgdir 'svgs' --nocdn
```


## Our team

At the moment we are *Skoltech DS MSc, 2019-2021* students.
* Artemenkov Aleksandr 
* Karpikov Igor
* Selikhanovych Daniil
