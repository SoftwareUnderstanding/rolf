# All colors matter
CS231N project, spring 2018

Vincent Billaut  
Matthieu de Rochemonteix  
Marc Thibault  

## Project overview

![churchill](img/churchill.png)  
Example taken [here](https://dribbble.com/shots/2122311-Photo-Colorization-Winston-Churchill).

Our goal is to tackle the problem of **colorization**, that
 is taking a gray-scale photograph and translating it to its colorized
version. This problem is particularly appealing to us, because it
involves a generative, creative work which is easily
visualizable and even fun. Our implementation features UNets.  Several articles tackle this problem, and one of them is Zhang et al. 2016 ([[1]](https://arxiv.org/abs/1603.08511)), and we focused on trying to reproduce part of their work and use their insight.  

One of the advantages of colorization is that any dataset of colored images that is available can be used, since  we only have to generate the corresponding gray-scale dataset. We used parts of both the [SUN  dataset](https://groups.csail.mit.edu/vision/SUN/) and [ImageNet](http://www.image-net.org/), restricted mostly to scenes of landscapes, coasts, beaches and forests. The goal of this simplification is to show that a simple, not too deep model can learn meaningful colorization patterns on rather small datasets, if the task is focused on certain types of images. In contrast with [1] who trained their (very) deep model on 1.5M+ images, and showcased impressive results on a very wide variety of scenes -- including cities, cars, humans, landscapes, and so on --, we show that it is possible to train a (much simpler) model on only about 13k images and still obtain reasonably good results on scenes representing nature.

We evaluate our model with a custom loss that takes into account the rarity of certain color tones and tries not to penalize too much rare colors, the goal being to achieve very colorful images.

One of the main advantages of this project is that the results are very easy to validate manually, the overall coherence of the color being easily perceptible with the human eye.

After training our ColorUNet model on about 13k data-augmented (which induces an additional 7x dilution) set, and obtaining satisfactory results, we tried colorizing videos. We began with a frame-by-frame approach, and then tried smoothing the predictions with an exponential kernel on the time dimension.

[1] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European Conference on Computer Vision. Springer, Cham, 2016.



![Predicted Video](img/predicted.gif)

Predicted video

![Predicted Video](img/predicted_smooth.gif)

Predicted video with smoothing

![Predicted Video](img/greyscale.gif)

Input video

![Predicted Video](img/true.gif)

Ground truth video


