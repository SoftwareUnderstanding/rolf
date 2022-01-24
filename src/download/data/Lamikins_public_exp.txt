## A collection of ML/DL experiments
Author: Darius Lam

I'm releasing several jupyter notebooks containing code I've written over the past several years. Below you will find descriptions of all of them (plus extras). Note that these were created for my own educational purposes.  Mathematical derivations are from sources including ML: APP by Murphy, Deep Learning by Goodfellow et al., and specified papers.

### Discriminant Analysis
In (Gaussian) discriminant analysis, we fit a generative classifier of class-conditional gaussians: <img src="https://tex.s2cms.ru/svg/p(x%7Cy%3Dc%2C%5Ctheta)%20%3D%20%5Cmathcal%7BN%7D(x%7C%5Cmu_c%2C%5CSigma_c)" alt="p(x|y=c,\theta) = \mathcal{N}(x|\mu_c,\Sigma_c)" />.  The distributions are fit using maximum likelihood, which has a simple solution for gaussians. Normally, we have one distribution per class and the covariance matrices are separate across classes.  However, in the case where the matrices are shared, such that for all classes, <img src="https://tex.s2cms.ru/svg/%5CSigma_c%20%3D%20%5CSigma" alt="\Sigma_c = \Sigma" />, we get Linear Discriminant Analysis (LDA).  Additionally, I implement a regularized version of LDA, such that the shared covariance matrix is diagonal. 

### Hidden Markov Models
An HMM is a Markov process with hidden states.  Our HMM consists of an observation model <img src="https://tex.s2cms.ru/svg/p(x_t%7Cz_t)" alt="p(x_t|z_t)" /> (the probability of an observation given a hidden state) and a transition model <img src="https://tex.s2cms.ru/svg/p(z_%7Bt%2B1%7D%2Cz_t)" alt="p(z_{t+1},z_t)" /> (the probability of the next hidden state given the current hidden state). Our hidden states are <img src="https://tex.s2cms.ru/svg/%5Ctextbf%7Bz%7D" alt="\textbf{z}" /> and observations <img src="https://tex.s2cms.ru/svg/%5Ctextbf%7Bx%7D" alt="\textbf{x}" />.  Then we have the corresponding joint distribution:

<img src="https://tex.s2cms.ru/svg/p(%5Ctextbf%7Bz%7D%2C%5Ctextbf%7Bx%7D)%20%3D%20p(%5Ctextbf%7Bz%7D)p(%5Ctextbf%7Bx%7D%7C%5Ctextbf%7Bz%7D)%20%3D%20p(z_1)%20%5Cprod_t%20p(z_t%7Cz_%7Bt-1%7D)%20%5Cprod_t%20p(%5Ctextbf%7Bx%7D_t%20%7Cz_t)" alt="p(\textbf{z},\textbf{x}) = p(\textbf{z})p(\textbf{x}|\textbf{z}) = p(z_1) \prod_t p(z_t|z_{t-1}) \prod_t p(\textbf{x}_t |z_t)" />

From this distribution we can perform difference inferences.  Filtering is the task of computing our belief state in an online fashion, while smoothing is computing our belief state offline, given all evidence. We can also use the Viterbi algorithm to find the most likely "path" of states.  I implement both the forwards, forwards-backwards, and Viterbi algorithm.

The HMM has a large number of applications in sequence modeling.  It preceded the RNN, with many concepts overlapping.

### Location Convolutions 
I recreated several of the experiments from Uber's CoordConv (Location Convolution) paper to examine the usefulness of this form of feature engineering.  The main contribution of the paper is to add additional channels at an uppermost CNN layer.  The additional channels simply broadcast x-axis, y-axis, and (optionally) radial coordinates, scaled to [-1,1].  The authors claim that by adding coordinate information to a CNN, it will drastically improve the ability of said CNN to learn tasks requiring spatial information.  In particular, they propose several tasks at which regular CNNs fail miserably but their "CoordConv" CNNs solve very quickly.  These are the Supervised Coordinate Classification, Supervised Rendering, and Unsupervised Density Learning task.  I experiment with the first two tasks in this notebook (the unsupervised learning experiment is in a different notebook).  True to their paper, I find that CoordConv performs much better at the tasks than regular CNNs.  

What's interesting about their idea to me is not so much its immediate content but rather their willingness to experiment with adding additional non-domain-specific data to existing models.  It makes me wonder what experimental impact other added features would have.  

source: https://arxiv.org/pdf/1807.03247.pdf

### Mixture Density Networks
I implemented the mixture density network originally proposed by Bishop. The network deals with a shortcoming of neural networks trained by Euclidean loss, namely that they learn to find the conditional average of the data (assuming gaussian distribution).  However, this is often suboptimal, for example in the case where our function is one-to-many and we need to output over a multimodal distribution.  The MDN resolves this issue.

### Markov Decision Processes & Dynamic Programming Solutions
MDPs are the backbone for almost all reinforcement learning problems today. 

### Maximum Likelihood & Maximum A Posteriori Estimation
This is an implementation of MLE and MAP estimation on a small 2D dataset.  The MLE and MAP estimation methods are fundamental to machine learning.

### Custom Multilayer Perception with Backpropagation
I wrote a very (very) simple but functional neural network package, using only numpy, implementing backpropagation and layer chaining.

### HD-GAN
Purpose: I experimented with generating 512x512 images from a custom dataset.  This was before NVIDIA's progressive growing paper.

Notes: Code unreleased.

Example output image:

<img src="HD_GAN.png" alt="hd-gan" width="512"/>

### Char-RNN Generating Ulysses
Purpose: I took one of my favorite texts and attempted to generate passages in its style.  
Notes: Code unreleased

Example output:
```
Bloomsalem behind Mays an Paddli. How petroster, respeeket.
Bad Met, my has pursuesalun. Suicides, Panawerat and Jimes, currop in the bontspeake thes in chokers,
black forth
doubles in, mannon, Salong. You remembers for the reformissity, appeant it
appaying caphond cracts! Spud a cold feet levoir her called two resone.
Mad Boat sheap and scullery hand gird today.
Ceeking her lapses snores he have passes of all your limpplace. Give and part is the blue it in black cap where him a brow, voolley, kneekneses,
apold, hat crickspeech anrual stands makeam. He shows, trunches firere? Kiscies, father, he steps
from the conseques and smack! Leopold!

(Midnight citism
at in the ladies round and Enthinta surches five Lordcacy sheeply toppies crop. Majer frock cheese
that, guaty appromicial raind home to be have. Bat appears laughshure. Vown, visances the Right. You warm.

BLOOM: (Squire match druepidants through that days snancous head sceptratung of pigles’ Palmos. Lae otterfully of the
Pharts! (To Metrs of sweet luttered Hungrog, conspaper of sir. Have you save you!
Sky back and year. A silk tram eyes and carefully. Slides girl, kings and whinis Mosst respected toft. I putts falling chuke of pastiller, plumb trop be hear gentlemen. He stops or not houses of the greated with accessory eyes and other makes draws
his enclubber, the fatchuia. All pulls her lapses watch in a roar
passiet and
reads and fart.

MRS YELVERTS CITITTHEREN: Mank hands reveated
hands past and call chatticial naw and kay, scar
```
