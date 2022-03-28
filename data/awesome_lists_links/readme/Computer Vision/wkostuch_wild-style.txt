# wild-style
Style transfer on images through machine learning.

# Overview
This project creates art with AI by using a number of variations on the method 
of deep learning style transfer.  In the file base_demo.py, an implementation 
is given of the style transfer algorithm given by Gatys et. al. in their seminal
paper, "A Neural Algorithm of Artistic Style".  The algorithm produces some
spectacular results:

## Images
### A Texas Starry Night
A Texas landscape rendered in the style of Van Gogh's "Starry Night".
> ![Texas Starry Night](https://github.com/wkostuch/wild-style/blob/master/images/results/Texas_night/texas_starry_night.png)

### Lakeside + Parable of the Sower
A lakeside combined with The Parable of the Sower produces a quite painterly
vista.
> ![Lakeside](https://github.com/wkostuch/wild-style/blob/master/images/content/lakeside.png),
> ![The Parable of the Sower](https://github.com/wkostuch/wild-style/blob/master/images/style/parable_of_sower.png),
> ![Lakeside + Parable of the Sower](https://github.com/wkostuch/wild-style/blob/master/images/results/lake%2Bparable_of_sower.png)

### Dog + Girl With a Mandolin
A dog rendered in the style of Picasso's "Girl with a Mandolin" produces a 
distinctly cubist dog.
> ![Dog](https://github.com/wkostuch/wild-style/blob/master/images/content/dog.png)
> ![Girl with a Mandolin](https://github.com/wkostuch/wild-style/blob/master/images/style/girl_with_mandolin.png)
> ![Cubist dog](https://github.com/wkostuch/wild-style/blob/master/images/results/picasso_dog_3.png)

# Results
Check out images/results for a sample of what the algorithm can produce!

# Directions
## Installation
Clone the repository to your computer.  From the root of the cloned repo, run the 
command `pip install -r requirements.txt` to install the required dependencies.

## Usage
To try it with your own images, edit the file paths for the content_path and 
`style_path` variables at the top of `base_demo.py` to point to images in your own 
file system, then run the file.

# References
## Papers
* A Neural Algorithm of Artistic Style: https://arxiv.org/abs/1508.06576
* Very Deep Convolutional Networks for Large-Scale Image Recognition: https://arxiv.org/abs/1409.1556
* An Improved Style Transfer Algorithm Using Feedforward Neural Network for 
Real-Time Image Conversion: https://www.mdpi.com/2071-1050/11/20/5673

## Articles
* Art & AI: The Logic Behind Deep Learning ‘Style Transfer’: https://medium.com/codait/art-ai-the-logic-behind-deep-learning-style-transfer-1f59f51441d1
* Neural Style Transfer: Creating Art with Deep Learning: https://blog.tensorflow.org/2018/08/neural-style-transfer-creating-art-with-deep-learning.html
