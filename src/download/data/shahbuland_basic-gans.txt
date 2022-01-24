# basic-gans

# Purpose of this repo
I'm using this repository to keep track of and document my exploration of GANs (Generative Adversarial Netorks),
I'm doing this to: 
1. Have a public record of some of the stuff I've been working on over the past year
2. Get better at documentation
3. Maybe help others who are trying to work on something similar learn implementation  
  
The datasets.py script is useful for loading a folder of images into python, feel free to put your
own photos into a dataset folder and try it
# Goals:
- Make DCGAN <sup>[1](#fn1)</sup>
- Make WGAN <sup>[2](#fn2)</sup>
- Make GP-WGAN <sup>[3](#fn3)</sup>
- Make Progressive <sup>[4](#fn4)</sup>

# References:
<a name="fn1">1</a>: https://arxiv.org/pdf/1511.06434.pdf  
<a name="fn2">2</a>: https://arxiv.org/pdf/1701.07875.pdf  
<a name="fn3">3</a>: https://arxiv.org/pdf/1704.00028.pdf  
<a name="fn4">4</a>: https://arxiv.org/pdf/1710.10196.pdf  

# Results
Left/Mid/Right : Real/DCGAN/GP-WGAN  
<img src="https://github.com/shahbuland/basic-gans/blob/master/results/dcgan/cars/real.png" alt="Real" height="250" width="250">
<img src="https://github.com/shahbuland/basic-gans/blob/master/results/dcgan/cars/fake.png" alt="Real" height="250" width="250">
<img src="https://github.com/shahbuland/basic-gans/blob/master/results/gpwgan/cars/fake.png" alt="Real" height="250" width="250">
