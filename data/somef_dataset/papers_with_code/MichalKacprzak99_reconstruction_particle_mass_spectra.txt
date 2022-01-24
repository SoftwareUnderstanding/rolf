# GANs in High Physics
## Project describtion 
Project was created during participation in the Krakow Applied Physicsand Computer Science Summer School ’20. The aim of the project was to test the latest GenerativeAdversarial Network (GAN) models for their application in simulating physical events;

## Participants 
 - Michał Kacprzak;
 - Paweł Kopciewicz as supervisor;
## Technologies
* Python 3.7


## Installation
    $ git clone https://github.com/MichalKacprzak99/reconstruction_particle_mass_spectra
    $ cd reconstruction_particle_mass_spectra
    $ sudo pip3 install -r requirements.txt
## Example how to use
1. Open files/main.py.
2. Create object of GAN.
3. Call class method "train".
```python
from GAN.gan import GAN
if __name__ == '__main__':
    gan = GAN()
    gan.train(30000)
```
## Implementations
### BGAN
Implementation of _Boundary-Seeking Generative Adversarial Networks_.

[Code](files/BGAN/bgan.py)

Paper: https://arxiv.org/abs/1702.08431
### DualGAN
Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.

[Code](files/DUALGAN/dualgan.py)

Paper: https://arxiv.org/abs/1704.02510
### GAN
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.

[Code](files/GAN/gan.py)

Paper: https://arxiv.org/abs/1406.2661
### WGAN
Implementation of _Wasserstein GAN_ (with DCGAN generator and discriminator).

[Code](files/WGAN/wgan.py)

Paper: https://arxiv.org/abs/1701.07875

## License
[MIT](https://choosealicense.com/licenses/mit/)


