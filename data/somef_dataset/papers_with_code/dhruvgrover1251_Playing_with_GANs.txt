# Playing_with_GANs
<p>This repository consist of implementation of DCGANs on 2 datasets (celeba and MNIST) for generating completely new looking images of faces and numbers respectively.</p>
<p> TO understand GANs here is the small summary of first paper(https://arxiv.org/pdf/1406.2661.pdf)</p>
<h3> Generative Adversarial Nets(Summary) </h3>
<h4> Introduction</h4>
<p> GANs take a game-theoretical approach: learn to generate from 2 player games to come over the problem of the intractable density function.
This model doesn’t assume explicit density function as in the case of VAEs.</p>
<h4> Adversarial networks </h4>
<p> In this, we have two types of networks,</p>
1. Generator network:<br>
<p>Try to fool discriminator by generating real looking images.</p>
<p>To learn the generator’s distribution p<sub>g</sub> over data x, we define a prior on input noise variables p<sub>g</sub>(z), then represent a mapping to data space as G(z; theta<sub>g</sub>), where G is a differentiable function represented by a multilayer perceptron with parameters g.</p>
2. Discriminator network: <br>
<p>Try to distinguish between real and fake images.</p>
<p>It is a second multilayer perceptron D(x;theta<sub> d</sub>) that outputs a single scalar. D(x) represents the probability that x came from the data rather than p<sub>g.</sub></p>

![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/GANs.PNG)

<h4>Mathematics behind above intuition</h4>

![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/GANs%202.PNG)
1. <p> Discriminator(theta<sub>d</sub> ) try to maximize objective function such that D(x) is close to 1(real) and D(G(z)) is close to 0(fake).</p>
<p>D is trained so that we get the maximum value of log(D<sub>theta d</sub>(x)) that is internally trying to label dataset images as real, also D is maximizing log(1-D<sub>theta d</sub>(G<sub>theta g</sub>(z)) this internally trying to label generated images as fake images. </p>
2. <p>Generator (theta<sub>g</sub> ) try to minimize objective such that D(G(Z)) is close to 1.</p>
<P>It is trying to minimize log(1-D<sub>theta d</sub>(G<sub> theta g </sub>(z)) thus setting g (through gradient descents) such that it can fool discriminator.
Rather than training G to minimize log(1 - D(G(z))) we can train G to maximize log(D(G(z)) because it provides steep gradients initially then it provides non steep.</p>
<h4>Implementation</h4>

![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/GANs%203.PNG)
<p>Note: Objective function obtain minimum for a given generator when p<sub>g</sub> = p<sub>data</sub></p>

<h4>Disadvantages</h4>
1.<p>There is no explicit representation of p<sub>g</sub> (x).</p>
2.<p>D must be well sync with G otherwise it could create problem.</p>
<p><b>Here paper summary of paper ends.</b></p>
<p>The notebook you see here is implemented in form of convolutional architecture popularly known as <b>DCGANs</b>.
<p> Here is basoc architecture of DCGANs.
  
  ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/DCGANS%20archi.PNG)
  
  <p>Here I implemented DCGANs on MNIST dataset and celeba(50000images due to lack of computational power)</p>
  
  <h3> Results of applying DCGANs on MNIST</h3>
  Generator loss and discriminator loss are as follows:<br>
  
  ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/MNIST%20Loss%20plot.PNG)<br>
  
  Training goes a follows:<br>
  
  ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/dcgan%20gif%20mnist.gif)<br>
  
  MNIST result after 50th epoch<br>
  ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/MNIST_result.png)
  
  <h3>Results of applying MNIST on celeba<h3>
  
  <b>Note: Here the model is not getting good results as it as facing mode collapse, so I am still improving this.</b><br>
  
  Discriminator and generator loss function are as follows: 
  
  ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/celeba_loss.png)<br>
  
   Training goes a follows:<br>
  
   ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/celeba_DCGANs.gif)<br>
   
   celeba result after 100th epoch<br>
   
   ![alt text](https://github.com/dhruvgrover1251/Playing_with_GANs/blob/master/celeba_results.png)<br>
   
   
   <b> There are more advances done for getting better results on human faces for that you can have a look at style GANs https://arxiv.org/pdf/1812.04948.pdf</b>
   
   

   
   
   
   



  
  

